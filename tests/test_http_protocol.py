"""Real h11 parser/ASGI tasks and asyncio timers; no mocked state machine."""
import asyncio

import h11
import pytest
import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState

from studio.http_protocol import SDVH11Protocol


class MemoryTransport(asyncio.Transport):
    def __init__(self):
        self.output = bytearray()
        self.closed = False
    def get_extra_info(self, name, default=None):
        return default
    def write(self, data):
        self.output.extend(data)
    def is_closing(self):
        return self.closed
    def close(self):
        self.closed = True
    def pause_reading(self):
        pass
    def resume_reading(self):
        pass


async def pipelined_stream(protocol_type, reproduce=False):
    loop = asyncio.get_running_loop()
    errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: errors.append(context))
    started, release = asyncio.Event(), asyncio.Event()
    async def app(scope, receive, send):
        request = await receive()
        assert request['type'] == 'http.request'
        if scope['path'] == '/first':
            await send({'type':'http.response.start','status':200,'headers':[(b'content-length',b'2')]})
            await send({'type':'http.response.body','body':b'ok'})
        else:
            await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'text/event-stream')]})
            await send({'type':'http.response.body','body':b'data: first\n\n','more_body':True})
            started.set()
            await release.wait()
            await send({'type':'http.response.body','body':b'data: last\n\n','more_body':False})
    config = uvicorn.Config(app, ws='none', lifespan='off', log_config=None,
                            timeout_keep_alive=.03)
    state, transport = ServerState(), MemoryTransport()
    protocol = protocol_type(config=config, server_state=state, app_state={}, _loop=loop)
    protocol.connection_made(transport)
    # Both requests arrive in one read. The second is processed by
    # on_response_complete, so no second data_received can cancel idle state.
    protocol.data_received(b'GET /first HTTP/1.1\r\nHost: localhost\r\n\r\n'
                           b'GET /stream HTTP/1.1\r\nHost: localhost\r\n\r\n')
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert protocol.conn.our_state is h11.SEND_BODY
        if reproduce:
            assert protocol.timeout_keep_alive_task is not None
            with pytest.raises(h11.LocalProtocolError, match='SEND_BODY'):
                protocol.timeout_keep_alive_handler()
            # Finish the deliberate failing baseline through the normal
            # disconnect path, avoiding an unrelated second error from send.
            protocol.connection_lost(None)
        else:
            assert protocol.timeout_keep_alive_task is None
            await asyncio.sleep(.07)
            assert not transport.closed
            assert protocol.conn.our_state is h11.SEND_BODY
            assert errors == []
        release.set()
        if state.tasks:
            await asyncio.wait_for(asyncio.gather(*tuple(state.tasks)), 1)
        if not reproduce:
            assert b'data: first' in transport.output and b'data: last' in transport.output
            assert state.total_requests == 2
            # Idle connections still time out; the fix preserves keep-alive
            # cleanup instead of disabling it for the entire server.
            assert protocol.timeout_keep_alive_task is not None
            await asyncio.sleep(.07)
            assert transport.closed
            assert protocol.conn.our_state is h11.CLOSED
            assert errors == []
    finally:
        release.set()
        protocol.connection_lost(None)
        loop.set_exception_handler(previous_handler)


@pytest.mark.skipif(uvicorn.__version__ != '0.22.0', reason='Regression reproduction targets the Python 3.7 Uvicorn pin')
def test_original_uvicorn_timer_enters_send_body_error():
    asyncio.run(pipelined_stream(H11Protocol, reproduce=True))


def test_buffered_sse_request_survives_idle_timeout_and_idle_cleanup_works():
    asyncio.run(pipelined_stream(SDVH11Protocol))
