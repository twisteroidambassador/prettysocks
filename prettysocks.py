#!/usr/bin/env python3

"""Simplistic SOCKS5 proxy with Happy Eyeballs for outgoing connections.

Two implementations of Happy Eyeballs can be used: either built-in with
Python 3.8.1 and up, or from the `async-stagger` module.
"""

"""
Copyright (C) 2018, 2020 twisteroid ambassador

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""


import asyncio
import collections
import contextlib
import enum
import errno
import ipaddress
import logging
import signal
import socket
import sys
import warnings
from functools import partial
from typing import Callable, Awaitable, Tuple, Union

try:
    import async_stagger
    from async_stagger.exceptions import HappyEyeballsConnectError
except ImportError:
    async_stagger = None
    class HappyEyeballsConnectError(Exception):
        pass


# ========== Configuration ==========

LISTEN_HOST = ['127.0.0.1', '::1']
LISTEN_PORT = 1080
LOGLEVEL = logging.INFO

# ========== Tunables ==========

"""When set to True, use Python's built-in Happy Eyeballs implementation
(Only available on Python 3.8.1 and up), which does not support asynchronous
address resolution. When set to False, use implementation in `async_stagger`
module."""
USE_BUILTIN_HAPPY_EYEBALLS = False

# The following specify Happy Eyeballs behavior. Refer to RFC 8305 for their
# definitions.
# https://tools.ietf.org/html/rfc8305#section-8
RESOLUTION_DELAY = 0.05  # seconds
FIRST_ADDRESS_FAMILY_COUNT = 1
CONNECTION_ATTEMPT_DELAY = 0.25  # seconds

# ==========


IPAddressType = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
HostType = Union[str, ipaddress.IPv4Address, ipaddress.IPv6Address]
ConnectFnType = Callable[[HostType, int],
                         Awaitable[Tuple[asyncio.StreamReader,
                                         asyncio.StreamWriter]]]
AcceptFnType = Callable[[asyncio.StreamReader,
                         asyncio.StreamWriter,
                         ConnectFnType],
                        Awaitable[Tuple[HostType,
                                        int,
                                        asyncio.StreamReader,
                                        asyncio.StreamWriter]]]
RelayFnType = Callable[[asyncio.StreamReader,
                        asyncio.StreamWriter,
                        asyncio.StreamReader,
                        asyncio.StreamWriter,
                        str],
                       Awaitable[None]]

INADDR_ANY = ipaddress.IPv4Address(0)


class BytesEnum(bytes, enum.Enum):
    pass


class SOCKS5AuthType(BytesEnum):
    NO_AUTH = b'\x00'
    GSSAPI = b'\x01'
    USERNAME_PASSWORD = b'\x02'
    NO_OFFERS_ACCEPTABLE = b'\xff'


class SOCKS5Command(BytesEnum):
    CONNECT = b'\x01'
    BIND = b'\x02'
    UDP_ASSOCIATE = b'\x03'


class SOCKS5AddressType(BytesEnum):
    IPV4_ADDRESS = b'\x01'
    DOMAIN_NAME = b'\x03'
    IPV6_ADDRESS = b'\x04'


class SOCKS5Reply(BytesEnum):
    SUCCESS = b'\x00'
    GENERAL_FAILURE = b'\x01'
    CONNECTION_NOT_ALLOWED_BY_RULESET = b'\x02'
    NETWORK_UNREACHABLE = b'\x03'
    HOST_UNREACHABLE = b'\x04'
    CONNECTION_REFUSED = b'\x05'
    TTL_EXPIRED = b'\x06'
    COMMAND_NOT_SUPPORTED = b'\x07'
    ADDRESS_TYPE_NOT_SUPPORTED = b'\x08'


class SOCKS5Acceptor:
    """Negotiate with downstream SOCKS5 clients."""
    _logger = logging.getLogger('socks5')

    def _map_exception_to_socks5_reply(self, exc: Exception) -> SOCKS5Reply:
        if isinstance(exc, HappyEyeballsConnectError):
            replies = map(self._map_exception_to_socks5_reply, exc.args[0])
            reply_counter = collections.Counter(replies)
            reply_counter.pop(SOCKS5Reply.GENERAL_FAILURE, None)
            if reply_counter:
                return reply_counter.most_common(1)[0][0]
            return SOCKS5Reply.GENERAL_FAILURE
        if isinstance(exc, socket.gaierror):
            return SOCKS5Reply.HOST_UNREACHABLE
        if isinstance(exc, TimeoutError):
            return SOCKS5Reply.TTL_EXPIRED
        if isinstance(exc, ConnectionRefusedError):
            return SOCKS5Reply.CONNECTION_REFUSED
        if isinstance(exc, OSError):
            if exc.errno == errno.ENETUNREACH:
                return SOCKS5Reply.NETWORK_UNREACHABLE
            elif exc.errno == errno.EHOSTUNREACH:
                return SOCKS5Reply.HOST_UNREACHABLE
            elif exc.errno == errno.ECONNREFUSED:
                return SOCKS5Reply.CONNECTION_REFUSED
            elif exc.errno == errno.ETIMEDOUT:
                return SOCKS5Reply.TTL_EXPIRED
            else:
                return SOCKS5Reply.GENERAL_FAILURE
        self._logger.warning('Unexpected exception', exc_info=exc)
        raise exc

    async def accept(
            self,
            dreader: asyncio.StreamReader,
            dwriter: asyncio.StreamWriter,
            connector: ConnectFnType,
    ) -> Tuple[
        HostType,
        int,
        asyncio.StreamReader,
        asyncio.StreamWriter,
    ]:
        """Negotiate with downstream SOCKS5 client.

        Accepts CONNECT command only, uses `connector` to connect to the
        upstream destination, and returns (dest_host, dest_port,
        upstream_reader, upstream_writer).
        """
        dname = repr(dwriter.get_extra_info('peername'))
        log_name = '{!s} <=> ()'.format(dname)
        try:
            buf = await dreader.readexactly(2)  # ver, number of auth methods
            if buf[0] != 5:
                raise ValueError('Invalid client request version')
            buf = await dreader.readexactly(buf[1])  # offered auth methods
            if SOCKS5AuthType.NO_AUTH not in buf:
                dwriter.write(b'\x05' + SOCKS5AuthType.NO_OFFERS_ACCEPTABLE)
                dwriter.write_eof()
                await dwriter.drain()
                raise ValueError(
                    'Client did not offer "no auth", offers: %r' % buf)
            dwriter.write(b'\x05' + SOCKS5AuthType.NO_AUTH)

            # client command
            buf = await dreader.readexactly(4)  # ver, cmd, rsv, addr_type
            if buf[0] != 5 or buf[2] != 0:
                raise ValueError('%s malformed SOCKS5 command'
                                 % log_name)
            cmd = SOCKS5Command(buf[1:2])
            addr_type = SOCKS5AddressType(buf[3:4])
            if addr_type is SOCKS5AddressType.IPV4_ADDRESS:
                uhost = ipaddress.IPv4Address(await dreader.readexactly(4))
            elif addr_type is SOCKS5AddressType.IPV6_ADDRESS:
                uhost = ipaddress.IPv6Address(await dreader.readexactly(16))
            elif addr_type is SOCKS5AddressType.DOMAIN_NAME:
                buf = await dreader.readexactly(1)  # address len
                uhost = (await dreader.readexactly(buf[0])).decode('utf-8')
                # Sometimes clients will pass in an IP address literal (such as
                # "127.0.0.1" as a host name. For example, Firefox does this if
                # network.proxy.socks_remote_dns is set to True.
                # However, we don't bother converting them into IPv(4|6)Address
                # objects here, since we will hand the address to connecting
                # functions that take strings anyway.
            else:
                raise ValueError('%s unsupported address type %r'
                                 % (log_name, addr_type))
            uport = int.from_bytes(await dreader.readexactly(2), 'big')
            log_name = '{!s} <=> ({!r}, {!r})'.format(dname, uhost, uport)
            self._logger.debug('%s parsed target address', log_name)
            if cmd is not SOCKS5Command.CONNECT:
                await self._reply(
                    dwriter, SOCKS5Reply.COMMAND_NOT_SUPPORTED, INADDR_ANY, 0)
                raise ValueError(
                    'Client command %r not supported' % cmd)
            self._logger.info('%s received CONNECT command', log_name)

            try:
                ureader, uwriter = await connector(uhost, uport)
            except Exception as e:
                self._logger.debug(
                    '%s Exception while connecting: %r', log_name, e)
                reply = self._map_exception_to_socks5_reply(e)
                await self._reply(dwriter, reply, INADDR_ANY, 0)
                raise

            sockname = uwriter.get_extra_info('sockname')
            bind_host = ipaddress.ip_address(sockname[0])
            bind_port = sockname[1]
            await self._reply(
                dwriter, SOCKS5Reply.SUCCESS, bind_host, bind_port)
            return uhost, uport, ureader, uwriter
        except asyncio.IncompleteReadError as e:
            raise ValueError('Client did not complete negotiation') from e

    async def _reply(
            self,
            dwriter: asyncio.StreamWriter,
            reply: SOCKS5Reply,
            host: HostType,
            port: int
    ) -> None:
        if isinstance(host, ipaddress.IPv4Address):
            b_addr = SOCKS5AddressType.IPV4_ADDRESS + host.packed
        elif isinstance(host, ipaddress.IPv6Address):
            b_addr = SOCKS5AddressType.IPV6_ADDRESS + host.packed
        else:
            b_addr = host.encode('idna')
            b_addr = (SOCKS5AddressType.DOMAIN_NAME
                      + len(b_addr).to_bytes(1, 'big') + b_addr)
        dwriter.write(b'\x05' + reply + b'\x00'
                      + b_addr + port.to_bytes(2, 'big'))
        if reply is not SOCKS5Reply.SUCCESS:
            dwriter.write_eof()
        await dwriter.drain()
        self._logger.debug('%r sent reply %s',
                           dwriter.get_extra_info('peername'), reply)


class Relayer:
    """Relay data between two (StreamReader, StreamWriter) pairs."""
    _logger = logging.getLogger('relay')

    def __init__(
            self,
            *,
            bufsize=2**13,
    ) -> None:
        self._bufsize = bufsize

    async def _relay_data_side(
            self,
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
            log_name: str,
    ) -> None:
        try:
            while True:
                buf = await reader.read(self._bufsize)
                if not buf:  # EOF
                    break
                self._logger.debug('%s passing data', log_name)
                writer.write(buf)
                await writer.drain()
            try:
                self._logger.debug('%s passing EOF', log_name)
                writer.write_eof()
                await writer.drain()
            except OSError as e:
                if e.errno == errno.ENOTCONN:
                    self._logger.debug(
                        '%s endpoint already closed when passing EOF', log_name)
                    return
                raise
        except Exception as e:
            self._logger.info('%s caught exception: %r', log_name, e)
            raise

    async def relay(
            self,
            dreader: asyncio.StreamReader,
            dwriter: asyncio.StreamWriter,
            ureader: asyncio.StreamReader,
            uwriter: asyncio.StreamWriter,
            uname: str,
    ) -> None:
        """Pass data from dreader to uwriter, and ureader to dwriter."""
        dname = repr(dwriter.get_extra_info('peername'))
        utask = asyncio.create_task(self._relay_data_side(
            dreader, uwriter, '{!s} --> {!s}'.format(dname, uname)))
        dtask = asyncio.create_task(self._relay_data_side(
            ureader, dwriter, '{!s} <-- {!s}'.format(dname, uname)))
        try:
            await asyncio.gather(utask, dtask)
        except:
            dtask.cancel()
            utask.cancel()
            raise


@contextlib.asynccontextmanager
async def closing_writer(writer: asyncio.StreamWriter):
    try:
        yield
    finally:
        writer.close()
        await writer.wait_closed()


async def handler(
        accept: AcceptFnType,
        connect: ConnectFnType,
        relay: RelayFnType,
        dreader: asyncio.StreamReader,
        dwriter: asyncio.StreamWriter
) -> None:
    """Main server handler."""
    logger = logging.getLogger('handler')
    dname = repr(dwriter.get_extra_info('peername'))
    log_name = '{!s} <=> ()'.format(dname)
    try:
        async with contextlib.AsyncExitStack() as stack:
            logger.debug('%s received connection', log_name)
            await stack.enter_async_context(closing_writer(dwriter))
            uhost, uport, ureader, uwriter = await accept(
                dreader, dwriter, connect)
            await stack.enter_async_context(closing_writer(uwriter))
            uname = '({!r}, {!r})'.format(uhost, uport)
            log_name = '{!s} <=> {!s}'.format(dname, uname)
            logger.info('%s relaying', log_name)
            await relay(dreader, dwriter, ureader, uwriter, uname)
            logger.info('%s done', log_name)
    except asyncio.CancelledError:
        logger.info('%s handler cancelled', log_name)
        raise
    except (OSError, ValueError, TimeoutError, HappyEyeballsConnectError) as e:
        logger.info('%s exception: %r', log_name, e)
    except Exception as e:
        logger.error('%s exception:', log_name, exc_info=e)


def sigterm_handler():
    logging.warning('Process received SIGTERM')
    sys.exit()


async def builtin_happy_eyeballs_connect(
        host: HostType,
        port: int,
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection(
        str(host),
        port,
        happy_eyeballs_delay=CONNECTION_ATTEMPT_DELAY,
        interleave=FIRST_ADDRESS_FAMILY_COUNT,
    )


async def async_stagger_connect(
        host: HostType,
        port: int,
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await async_stagger.open_connection(
        str(host),
        port,
        delay=CONNECTION_ATTEMPT_DELAY,
        interleave=FIRST_ADDRESS_FAMILY_COUNT,
        async_dns=True,
        resolution_delay=RESOLUTION_DELAY,
        detailed_exceptions=True,
    )


async def amain():
    loop = asyncio.get_event_loop()
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGTERM, sigterm_handler)
    acceptor = SOCKS5Acceptor()
    relayer = Relayer()
    if USE_BUILTIN_HAPPY_EYEBALLS:
        connector = builtin_happy_eyeballs_connect
    else:
        if async_stagger is None:
            raise ImportError(
                'async_stagger module is required, but cannot be imported. '
                'To use without async_stagger, set '
                'USE_BUILTIN_HAPPY_EYEBALLS = True in code.'
            )
        connector = async_stagger_connect
    proxy_handler = partial(
        handler,
        acceptor.accept,
        connector,
        relayer.relay,
    )
    server = await asyncio.start_server(proxy_handler, LISTEN_HOST, LISTEN_PORT)
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        server.close()
        await server.wait_closed()


def main():
    rootlogger = logging.getLogger()
    rootlogger.setLevel(LOGLEVEL)
    stream_formatter = logging.Formatter('%(levelname)-8s %(name)s %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    rootlogger.addHandler(stream_handler)
    logging.captureWarnings(True)
    warnings.filterwarnings('always')

    try:
        asyncio.run(amain())
    except (KeyboardInterrupt, SystemExit) as e:
        logging.warning('Caught %r', e)


if __name__ == '__main__':
    main()
