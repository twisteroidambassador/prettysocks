# The IPv6 link-local literal hack fork

This fork of `prettysocks` is a workaround for browsers not supporting
literal IPv6 link-local addresses.

These links document the problem:

<https://ungleich.ch/u/blog/ipv6-link-local-support-in-browsers/>

<https://bugzilla.mozilla.org/show_bug.cgi?id=700999>

The TL;DR is, browsers like Firefox and Chrome do not support IPv6 address
literals that have a scope ID in them, which means one cannot browse to
link local IPv6 addresses.

This fork of `prettysocks.py`, apart from being a generic SOCKS5 proxy
server, also has one additional feature: it will replace [Microsoft's "magic
escape sequence" ipv6-literal.net host names](https://devblogs.microsoft.com/oldnewthing/20100915-00/?p=12863)
to the corresponding literal address.

So, say you want to visit `http://[fe80::1ff:fe23:4567:890a%3]` in Firefox.
Typing that address directly into the address bar will not bring you there, 
instead Firefox will try to search that in your default search engine.
Here's how to use `prettysocks` to achieve that:

* Run `prettysocks.py` (Python 3.8.1+ required, or 3.7+ if the
[`async-stagger` module][3] is installed and the script edited accordingly).
It will start a SOCKS5 proxy listening on `::1` port 1080.

* Configure Firefox to use a SOCKS5 proxy on `::1` port 1080. Don't worry,
`prettysocks` will pass through normal requests just fine, but you can also
use a proxy manager add-on to only use this proxy for `*.ipv6-literal.net`.

* Take the IPv6 address literal `fe80::1ff:fe23:4567:890a%3`,
replace colons `:` with dashes `-`, replace the percent mark `%` with the
letter `s`, and add a suffix `.ipv6-literal.net`.
The result is `fe80--1ff-fe23-4567-890as3.ipv6-literal.net`.

* Tack on the protocol part, any port numbers and paths you need, and type
the URL `http://fe80--1ff-fe23-4567-890as3.ipv6-literal.net` in the address
bar. Voila!

This should work with any software that can use SOCKS5 proxy servers.


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

The `prettysocks` script itself requires Python 3.7 or higher. 
(It should be possible to make it run on Python 3.6 with minimal modifications.)

If using built-in Happy Eyeballs implementation, Python 3.8.1 or higher is
required.
(The implementation was added in 3.8.0, but unfortunately a serious bug was only
fixed in 3.8.1.)

If using `async-stagger`, the module needs to be installed.

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