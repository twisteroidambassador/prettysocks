"""Simplistic SOCKS5 proxy implementing Happy Eyeballs for outgoing connections.

Implements the following features:

* Asynchronous hostname resolution: IPv6 and IPv4 resolution in parallel
* Address family interleaving
* Staggered connection attempts with constant delay
"""

# Copyright 2018 twisteroid ambassador
# Licensed under GNU GPL v3

import asyncio
import builtins
import collections
import contextlib
import enum
import errno
import ipaddress
import itertools
import logging
import signal
import socket
import sys
import warnings
from functools import partial
from typing import Optional, Callable, Iterable, Awaitable, Tuple, Any, List, \
    Union, AsyncIterable


# ========== Configuration ==========

LISTEN_HOST = ['127.0.0.1', '::1']
LISTEN_PORT = 1080
LOGLEVEL = logging.INFO

# ========== Tunables ==========

# Enable if it's necessary to resolve IP address literals ("127.0.0.1", "::1")
# using getaddrinfo().
USE_GETADDRINFO_ON_IP_ADDRESSES = False

# The following specify Happy Eyeballs behavior. Refer to RFC 8305 for their
# definitions.
# https://tools.ietf.org/html/rfc8305#section-8
RESOLUTION_DELAY = 0.05  # seconds
FIRST_ADDRESS_FAMILY_COUNT = 1
CONNECTION_ATTEMPT_DELAY = 0.25  # seconds

# ==========


IPAddressType = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
HostType = Union[str, ipaddress.IPv4Address, ipaddress.IPv6Address]
AddrInfoType = Tuple[int, int, int, str, Tuple]
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


# We need to use aiter() and anext() for async iterators, similar to iter() and
# next() for regular iterators. These were promised for Python 3.7, but hasn't
# yet landed as of Python 3.7 rc1. The following polyfills are adapted from the
# patch in the bpo entry linked below.
# https://bugs.python.org/issue31861
# TODO: if aiter and anext lands in the operator module, test and import

if not hasattr(builtins, 'aiter'):
    def aiter(obj):
        if not isinstance(obj, collections.abc.AsyncIterable):
            raise TypeError('%r object is not an asynchronous iterable'
                            % (type(obj).__name__,))
        return obj.__aiter__()

if not hasattr(builtins, 'anext'):
    def anext(iterator):
        if not isinstance(iterator, collections.abc.AsyncIterator):
            raise TypeError('%r object is not an asynchronous iterator'
                            % (type(iterator).__name__,))
        return iterator.__anext__()


def roundrobin(*in_iters: Iterable, _sentinel=object()) -> Iterable:
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    return (a for a in itertools.chain.from_iterable(itertools.zip_longest(
                *in_iters, fillvalue=_sentinel)) if a is not _sentinel)


async def staggered_race(
        coro_fns: AsyncIterable[Callable[[], Awaitable]],
        delay: Optional[float],
        *,
        loop: asyncio.AbstractEventLoop = None,
) -> Tuple[
    Any,
    Optional[int],
    List[Optional[Exception]]
]:
    """Run coroutines with staggered start times, take first to finish.

    The next coroutine is started after the previous one has been running for
    `delay` seconds, or when the previous one fails.

    Args:
        coro_fns: async iterable of coroutine functions to run. A coroutine
            that returns is deemed successful; one that raises an exception
            (including CancelledError) is deemed failed. There must be no more
            than one coroutine that succeeds. As soon as one succeeds, all
            others are cancelled.
        delay: amount of time to wait between starting coroutines in parallel.

    Returns: tuple (winner_result, winner_index, exceptions) where:
        * winner_result is the return value of the successful coroutine, or
            None if no coroutine succeeds.
        * winner_index is the index of the successful coroutine in coro_fns, or
            None if no coroutine succeeds.
        * exceptions is a list of exceptions raised by started coroutines. The
            entry for the successful coroutine is None.
    """
    loop = loop or asyncio.get_event_loop()
    aiter_coro_fns = aiter(coro_fns)
    winner_result = None
    winner_index = None
    exceptions = []
    tasks = []

    async def run_one_coro(previous_failed: Optional[asyncio.Event],
                           this_index: int = 0) -> None:
        # Wait for the previous task to finish, or for delay seconds
        if previous_failed is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(previous_failed.wait(), delay)
        # Get the coroutine to run
        try:
            coro_fn = await anext(aiter_coro_fns)
        except StopAsyncIteration:
            return
        # Start task that will run the next coroutine
        this_failed = asyncio.Event()
        next_task = loop.create_task(run_one_coro(this_failed, this_index+1))
        tasks.append(next_task)
        assert len(tasks) == this_index + 2
        # Prepare place to put this coroutine's exceptions if not won
        exceptions.append(None)
        assert len(exceptions) == this_index + 1

        try:
            result = await coro_fn()
        except Exception as e:
            exceptions[this_index] = e
            this_failed.set()  # Kickstart the next coroutine
        else:
            # Store winner's results
            nonlocal winner_index, winner_result
            assert winner_index is None
            winner_index = this_index
            winner_result = result
            # Cancel all other tasks. We take care to not cancel the current
            # task as well. If we do so, then since there is no `await` after
            # here and CancelledError are usually thrown at one, we will
            # encounter a curious corner case where the current task will end
            # up as done() == True, cancelled() == False, exception() ==
            # asyncio.CancelledError.
            # https://bugs.python.org/issue30048
            for i, t in enumerate(tasks):
                if i != this_index:
                    t.cancel()

    first_task = loop.create_task(run_one_coro(None))
    tasks.append(first_task)
    try:
        # Wait for a growing list of tasks to all finish: poor man's version of
        # curio's TaskGroup or trio's nursery
        done_count = 0
        while done_count != len(tasks):
            done, _ = await asyncio.wait(tasks)
            done_count = len(done)
            # If run_one_coro raises an unhandled exception, it's probably a
            # programming error, and I want to see it.
            if __debug__:
                for d in done:
                    if d.done() and not d.cancelled() and d.exception():
                        raise d.exception()
        return winner_result, winner_index, exceptions
    finally:
        # Make sure no tasks are left running if we leave this function
        for t in tasks:
            t.cancel()


class MultipleConnectionError(ConnectionError):
    """Container for multiple exceptions when connecting.

    args should be: (msg, [exc1, exc2, ...])
    """
    pass


class SOCKS5Acceptor:
    """Negotiate with downstream SOCKS5 clients."""
    _logger = logging.getLogger('socks5')

    def _map_exception_to_socks5_reply(self, exc: Exception) -> SOCKS5Reply:
        if isinstance(exc, MultipleConnectionError):
            replies = map(self._map_exception_to_socks5_reply, exc.args[1])
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
        if isinstance(exc, ConnectionError):
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
            assert buf[0] == 5
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
                # network.proxy.socks_remote_dns is set to True. We convert
                # those to actual IP addresses here.
                with contextlib.suppress(ValueError):
                    uhost = ipaddress.ip_address(uhost)
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

            try:
                sockname = uwriter.get_extra_info('sockname')
                await self._reply(
                    dwriter, SOCKS5Reply.SUCCESS, sockname[0], sockname[1])
                return uhost, uport, ureader, uwriter
            except:
                uwriter.transport.abort()
                raise
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


class AsyncResolveIter:
    """Async iterator that resolves a host name and yields its addresses."""
    _logger = logging.getLogger('aresolve')

    def __init__(
            self,
            host: HostType,
            port: int,
            *,
            resolution_delay: float = RESOLUTION_DELAY,
            first_address_family_count: int = FIRST_ADDRESS_FAMILY_COUNT,
            proto: int = 0,
            flags: int = 0,
            loop: asyncio.AbstractEventLoop = None,
    ) -> None:
        """Create async iterator instance and start resolution tasks.

        Args:
            host, port: hostname and port number to resolve.
            resolution_delay, first_address_family_count: controls Happy
                Eyeballs behavior. Refer to RFC 8305 for details.
            proto, flags: passed to getaddrinfo().
        """
        self._resolution_delay = resolution_delay
        self._interleave = first_address_family_count
        self._loop = loop or asyncio.get_event_loop()

        self._can_yield = asyncio.Event(loop=self._loop)
        self._can_yield.clear()
        self._addrinfos = collections.deque()
        self._yielded = 0
        self._ipv4_done = self._ipv6_done = False
        if isinstance(host, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            if not USE_GETADDRINFO_ON_IP_ADDRESSES:
                self._logger.debug(
                    'IP address %r does not need resolution', host)
                self._addrinfos.append((
                    socket.AF_INET6 if host.version == 6 else socket.AF_INET,
                    socket.SOCK_STREAM,
                    proto,
                    '',
                    (host.compressed, port),
                ))
                self._ipv4_done = self._ipv6_done = True
                self._can_yield.set()
                return
            else:
                host = host.compressed
        self._ipv6_task = loop.create_task(self._resolve_ipv6(
            host, port, proto, flags))
        self._ipv4_task = loop.create_task(self._resolve_ipv4(
            host, port, proto, flags))

    async def _resolve_ipv6(
            self,
            host: str,
            port: int,
            proto: int,
            flags: int,
    ) -> None:
        try:
            self._logger.debug('Resolving IPv6 address of %s', host)
            ipv6_addrinfos = collections.deque(await self._loop.getaddrinfo(
                host, port, family=socket.AF_INET6, type=socket.SOCK_STREAM,
                proto=proto, flags=flags))
            self._logger.debug('IPv6 addresses of %r: %r', host, ipv6_addrinfos)
            if not self._addrinfos:
                self._addrinfos.extend(ipv6_addrinfos)
            else:
                ipv4_addrinfos = self._addrinfos.copy()
                self._addrinfos.clear()
                for _ in range(self._interleave):
                    self._addrinfos.append(ipv6_addrinfos.popleft())
                self._addrinfos.extend(roundrobin(
                    ipv4_addrinfos, ipv6_addrinfos))
            self._logger.debug('Current address list: %r', self._addrinfos)
        except socket.gaierror as e:
            self._logger.info(
                'getaddrinfo() error resolving IPv6 address of %s: %r', host, e)
        except Exception:
            self._logger.error(
                'Exception resolving IPv6 address of %s', host, exc_info=True)
        finally:
            self._can_yield.set()
            self._ipv6_done = True

    async def _resolve_ipv4(
            self,
            host: str,
            port: int,
            proto: int,
            flags: int,
    ) -> None:
        try:
            self._logger.debug('Resolving IPv4 address of %s', host)
            ipv4_addrinfos = await self._loop.getaddrinfo(
                host, port, family=socket.AF_INET, type=socket.SOCK_STREAM,
                proto=proto, flags=flags)
            self._logger.debug('IPv4 addresses of %r: %r', host, ipv4_addrinfos)
            if not self._addrinfos:
                self._addrinfos.extend(ipv4_addrinfos)
            else:
                ipv6_addrinfos = self._addrinfos.copy()
                self._addrinfos.clear()
                for _ in range(self._interleave - self._yielded):
                    self._addrinfos.append(ipv6_addrinfos.popleft())
                self._addrinfos.extend(roundrobin(
                    ipv4_addrinfos, ipv6_addrinfos))
            self._logger.debug('Current address list: %r', self._addrinfos)
            self._ipv4_done = True
            if not self._ipv6_done:
                self._logger.debug(
                    'IPv6 resolution not done yet, wait %f before yielding '
                    'addresses', self._resolution_delay)
                await asyncio.sleep(self._resolution_delay)
        except socket.gaierror as e:
            self._logger.info(
                'getaddrinfo() error resolving IPv4 address of %s: %r', host, e)
        except Exception:
            self._logger.error(
                'Exception resolving IPv6 address of %s', host, exc_info=True)
        finally:
            self._can_yield.set()
            self._ipv4_done = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            await self._can_yield.wait()
            if self._addrinfos:
                addrinfo = self._addrinfos.popleft()
                self._logger.debug('Yielding %r', addrinfo)
                self._yielded += 1
                return addrinfo
            if self._ipv4_done and self._ipv6_done:
                raise StopAsyncIteration
            self._can_yield.clear()


class StatelessHappyEyeballsConnector:
    """Make outgoing connections using Happy Eyeballs algorithm."""
    _logger = logging.getLogger('he-stateless')

    def __init__(
            self,
            delay: float = CONNECTION_ATTEMPT_DELAY,
            *,
            loop: asyncio.AbstractEventLoop = None
    ) -> None:
        self._delay = delay
        self._loop = loop or asyncio.get_event_loop()

    async def _connect_sock(self, addr_info: AddrInfoType) -> socket.socket:
        """Create, bind and connect one socket."""
        family, type_, proto, _, address = addr_info
        self._logger.debug('Connecting to %r', address)
        sock = socket.socket(family=family, type=type_, proto=proto)
        try:
            sock.setblocking(False)
            await self._loop.sock_connect(sock, address)
            self._logger.debug('Connection to %r succeeded', address)
            return sock
        except:
            sock.close()
            raise

    async def connect(
            self,
            host: HostType,
            port: int,
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Make outgoing connection."""
        addrinfo_iter = AsyncResolveIter(host, port, loop=self._loop)

        async def coro_iter():
            async for addrinfo in addrinfo_iter:
                yield partial(self._connect_sock, addrinfo)

        sock, _, exceptions = await staggered_race(
            coro_iter(), self._delay, loop=self._loop)
        if not sock:
            if not exceptions:
                raise socket.gaierror('getaddrinfo() returned empty result')
            if len(exceptions) == 1:
                raise exceptions[0]
            else:
                raise MultipleConnectionError('Multiple exceptions', exceptions)
        try:
            return await asyncio.open_connection(sock=sock, loop=self._loop)
        except:
            sock.close()
            raise


class Relayer:
    """Relay data between two (StreamReader, StreamWriter) pairs."""
    _logger = logging.getLogger('relay')

    def __init__(
            self,
            *,
            loop: asyncio.AbstractEventLoop = None,
            bufsize=2**13,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
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
        utask = self._loop.create_task(self._relay_data_side(
            dreader, uwriter, '{!s} --> {!s}'.format(dname, uname)))
        dtask = self._loop.create_task(self._relay_data_side(
            ureader, dwriter, '{!s} <-- {!s}'.format(dname, uname)))
        try:
            await asyncio.gather(utask, dtask, loop=self._loop)
        except:
            dwriter.transport.abort()
            uwriter.transport.abort()
            dtask.cancel()
            utask.cancel()
            raise


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
        with contextlib.ExitStack() as stack:
            logger.debug('%s received connection', log_name)
            stack.enter_context(contextlib.closing(dwriter))
            uhost, uport, ureader, uwriter = await accept(
                dreader, dwriter, connect)
            stack.enter_context(contextlib.closing(uwriter))
            uname = '({!r}, {!r})'.format(uhost, uport)
            log_name = '{!s} <=> {!s}'.format(dname, uname)
            logger.info('%s relaying', log_name)
            await relay(dreader, dwriter, ureader, uwriter, uname)
            logger.info('%s done', log_name)
    except asyncio.CancelledError:
        logger.info('%s handler cancelled', log_name)
    except (OSError, ValueError, TimeoutError) as e:
        logger.info('%s exception: %r', log_name, e)
    except Exception as e:
        logger.error('%s exception:', log_name, exc_info=e)


def sigterm_handler():
    logging.warning('Process received SIGTERM')
    sys.exit()


def main():
    rootlogger = logging.getLogger()
    rootlogger.setLevel(LOGLEVEL)
    stream_formatter = logging.Formatter('%(levelname)-8s %(name)s %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    rootlogger.addHandler(stream_handler)
    logging.captureWarnings(True)
    warnings.filterwarnings('always')

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, sigterm_handler)
    connector = StatelessHappyEyeballsConnector(loop=loop)
    acceptor = SOCKS5Acceptor()
    relayer = Relayer(loop=loop)
    proxy_handler = partial(handler,
                            acceptor.accept,
                            connector.connect,
                            relayer.relay)
    loop.run_until_complete(asyncio.start_server(
        proxy_handler, LISTEN_HOST, LISTEN_PORT))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logging.warning('Caught KeyboardInterrupt')


if __name__ == '__main__':
    main()
