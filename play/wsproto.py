"""A minimal RFC 6455 WebSocket, enough for this app and nothing more.

Why not the `websockets` package: it is asyncio, and this server is a
thread-per-connection ThreadingHTTPServer on a single port that Cloudflare
already routes to via an Origin Rule. A second asyncio server would need its
own port, its own Origin Rule and its own firewall hole. Upgrading in place
costs ~150 lines and no new moving parts.

We control both endpoints, so this implements only what they use: text frames,
close, ping/pong, and continuation. Binary and extensions are not supported and
say so rather than silently mis-framing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import threading

GUID = "258EAFA5-E914-47DA-95CA-5AB0DC85B11F"

OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


def accept_key(key: str) -> str:
    return base64.b64encode(
        hashlib.sha1((key + GUID).encode()).digest()).decode()


class WSError(Exception):
    pass


class WebSocket:
    """One connection. Sends are locked; a socket is not safe for concurrent writes.

    The lock matters here specifically: results are pushed from WORKER threads
    into CLIENT sockets, so two workers finishing at once would otherwise
    interleave two frames on one wire and corrupt both.
    """

    def __init__(self, sock, max_bytes=8 * 1024 * 1024):
        self.sock = sock
        self.max_bytes = max_bytes
        self._send_lock = threading.Lock()
        self.closed = False

    # ---- low level ----------------------------------------------------
    def _read_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise WSError("peer closed")
            buf += chunk
        return buf

    def _send_frame(self, opcode, payload=b""):
        if self.closed:
            raise WSError("closed")
        n = len(payload)
        if n < 126:
            hdr = struct.pack("!BB", 0x80 | opcode, n)
        elif n < (1 << 16):
            hdr = struct.pack("!BBH", 0x80 | opcode, 126, n)
        else:
            hdr = struct.pack("!BBQ", 0x80 | opcode, 127, n)
        with self._send_lock:
            self.sock.sendall(hdr + payload)      # server frames are unmasked

    # ---- public -------------------------------------------------------
    def send_json(self, obj):
        self._send_frame(OP_TEXT, json.dumps(obj).encode())

    def ping(self):
        self._send_frame(OP_PING, b"")

    def recv_json(self, timeout=None):
        """Next application message, or None on ping/pong. Raises on close."""
        self.sock.settimeout(timeout)
        data, opcode = b"", None
        while True:
            b1, b2 = self._read_exact(2)
            fin, op = b1 & 0x80, b1 & 0x0F
            masked, ln = b2 & 0x80, b2 & 0x7F
            if ln == 126:
                ln = struct.unpack("!H", self._read_exact(2))[0]
            elif ln == 127:
                ln = struct.unpack("!Q", self._read_exact(8))[0]
            if ln > self.max_bytes:
                # A frame this large is either a bug or an attempt to exhaust
                # memory on a 1 GB box. Refuse rather than allocate.
                self.close(1009)
                raise WSError("frame too large")
            mask = self._read_exact(4) if masked else None
            payload = self._read_exact(ln) if ln else b""
            if mask:
                payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))

            if op == OP_CLOSE:
                self.close()
                raise WSError("peer closed")
            if op == OP_PING:
                self._send_frame(OP_PONG, payload)
                return None
            if op == OP_PONG:
                return None
            if op == OP_BIN:
                self.close(1003)
                raise WSError("binary not supported")
            if op in (OP_TEXT, OP_CONT):
                if opcode is None:
                    opcode = op
                data += payload
                if fin:
                    try:
                        return json.loads(data.decode())
                    except Exception as e:                 # noqa: BLE001
                        raise WSError(f"bad json: {e}")

    def close(self, code=1000):
        if self.closed:
            return
        self.closed = True
        try:
            self._send_frame(OP_CLOSE, struct.pack("!H", code))
        except Exception:                                   # noqa: BLE001
            pass
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:                                   # noqa: BLE001
            pass
        try:
            self.sock.close()
        except Exception:                                   # noqa: BLE001
            pass


def upgrade(handler):
    """Complete the handshake on a BaseHTTPRequestHandler. Returns a WebSocket."""
    key = handler.headers.get("Sec-WebSocket-Key")
    if not key or "websocket" not in (handler.headers.get("Upgrade") or "").lower():
        return None
    resp = ("HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n")
    handler.wfile.write(resp.encode())
    handler.wfile.flush()
    return WebSocket(handler.connection)
