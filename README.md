# prettysocks

*A proxy server that makes your eyeballs happy*

`prettysocks` is a simplistic, dual-stack friendly SOCKS5 proxy server
that implements Happy Eyeballs for outgoing connections.

## Motivation

Pretty much all modern web browsers support Happy Eyeballs, so as to
be a good Internet citizen. However, the same can't be said for many
proxy server software.

Install and configure [Dante proxy server][2]
on a dual-stack server, configure your browser to use it, and hit up
<https://ipv6-test.com/>. In **Browser - Fallback** you will see a big
red **"no"**, and the site will advise you to
**Upgrade your web browser**.
Of course, it's not your browser at fault, but the proxy server's, for using
a simple connection algorithm that has not kept up with the times.

The purpose of `prettysocks` is to be a dual-stack friendly proxy server,
so that browsers behind it still get the benefit of
Happy Eyeballs and all the goodness of IPv6.

## Features

Happy Eyeballs v2 is specified in [RFC 8305][1]. There are two implementations
of Happy Eyeballs that `prettysocks` can use: Python asyncio built-in, or
the [`async-stagger` module][3]. Both provide the following features:

* Interleaving addresses by family: connection attempts will use IPv6
  and IPv4 addresses alternatively.

* Parallel connection attempts: if a connection attempt to one address
  does not complete quickly, another attempt to the next address is
  started in parallel. (Delay time between attempts is fixed.)

In addition, `async-stagger` also supports:

* Asynchronous hostname resolution: IPv6 and IPv4 addresses for a given
  hostname are resolved in parallel, and connection attempts start
  before all addresses are resolved.

`prettysocks` does not directly implement DNS64 and NAT64. If the
operating system supports synthesizing IPv6 addresses via
`getaddrinfo()`, it might be supported automatically, or it might not.

### Non-features

`prettysocks` does not support, and has no plan to support in the
future, these features:

* Any SOCKS5 authentication methods except "no auth"

* SOCKS5 commands other than CONNECT, namely BIND and UDP-ASSOCIATE

* Any access control or filtering

* Any built-in provision of running as a daemon, detaching from the
  current terminal, etc.

## Requirements

The `prettysocks` script itself requires Python 3.11 or higher. 

If using `async-stagger`, a version >= 0.4.0 must be installed.

(For Python >= 3.7, < 3.11, or `async-stagger` < 0.4.0, use commits up to
11c646a2f4275025d4637d9c1f1329e1fc5d7e70 )

`prettysocks` is written for running on Linux, but should
work on other operating systems.
It is most useful on a server with both IPv6 and IPv4 Internet
connectivity, however there should be no harm in running it on a
single-stack server.

## Usage

Just run it. By default it listens on 127.0.0.1 and ::1 port 1080, and
prints logging output to `STDERR`. To change configuration options,
simply edit the script: all options are near the beginning.

To enjoy the benefits of Happy Eyeballs, the client software should be
configured to pass the host name to the proxy server, instead of doing
its own hostname resolution and passing IP addresses. For example, in
Firefox, **"Proxy DNS when using SOCKS v5"** must be turned on.

If running as a service is desired, write a simple systemd unit
file for it.

## Remarks

This project started as an attempt to implement Happy Eyeballs entirely
in stock `asyncio`, inspired by [the implementation in `trio`.][4]
I used the same logic for multiple projects, each time writing it from scratch,
before finally making a standalone module
<https://github.com/twisteroidambassador/async_stagger>, and eventually
contributing a simpler version into Python's standard library itself.

Afterwards, `prettysocks` lay forgotten for quite a while. I am now updating
it to use these latest implementations of Happy Eyeballs, primarily as
a testing tool for these implementations, but also in the hope that it can
be useful as an actual proxy server.

## License

`prettysocks` is licensed under GNU GPL v3.


[1]: https://tools.ietf.org/html/rfc8305
[2]: https://www.inet.no/dante/
[3]: https://pypi.org/project/async-stagger/
[4]: https://github.com/python-trio/trio/pull/145/files