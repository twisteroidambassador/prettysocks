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
<http://ipv6-test.com/>. In **Browser - Fallback** you will see a big
red **"no"**, and the site will advise you to
**Upgrade your web browser**.
Of course, it's not your browser at fault, but the proxy server.

That's why I have written `prettysocks`, so that even when using a
proxy server, my browser still get the benefit of
Happy Eyeballs and all the goodness of IPv6.

## Features

`prettysocks` implements these Happy Eyeballs v2 features specified in
[RFC 8305][1]:

* Asynchronous hostname resolution: IPv6 and IPv4 addresses for a given
  hostname are resolved in parallel, and connection attempts start
  before all addresses are resolved.

* Interleaving addresses by family: connection attempts will use IPv6
  and IPv4 addresses alternatively.

* Parallel connection attempts: if a connection attempt to one address
  does not complete quickly, another attempt to the next address is
  started in parallel. (Delay time between attempts is fixed.)

`prettysocks` does not directly implement DNS64 and NAT64. If the
operating system supports synthesizing IPv6 addresses via
`getaddrinfo()`, a configuration option can be turned on to utilize
that feature, by resolving all received IP address literals using
`getaddrinfo()`.

### Non-features

`prettysocks` does not support, and has no plan to support in the
future, these features:

* Any SOCKS5 authentication methods except "no auth"

* SOCKS5 commands other than CONNECT, namely BIND and UDP-ASSOCIATE

* Any access control or filtering

* Any built-in provision of running as a daemon, detaching from the
  current terminal, etc.

## Requirements

`prettysocks` requires Python 3.6 or higher. It does not have any
3rd-party dependencies. It is written for running on Linux, but should
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

## Future plans

The next step is to use historical RTT data in address sorting and
connection attempt delay calculations. This has lead to quite a deep
rabbit hole, with `ip tcp_metrics`, generic netlink, Linux kernel source
code, etc etc.

If a version utilizing historical RTT is written, it will most likely
be Linux-only.

## License

`prettysocks` is licensed under GNU GPL v3.


[1]: https://tools.ietf.org/html/rfc8305
[2]: https://www.inet.no/dante/