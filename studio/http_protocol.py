"""Keep Uvicorn's idle timer separate from active/pipelined SSE responses.

Python 3.7 uses Uvicorn 0.22. Its response-complete callback arms a keep-alive
timer before dispatching an already buffered HTTP request. That request does
not pass through data_received again, so the old timer can fire in SEND_BODY.

Current upstream cancels the timer when processing h11.Request:
https://github.com/Kludex/uvicorn/blob/main/uvicorn/protocols/http/h11_impl.py
This small subclass supplies that lifecycle correction without copying the
version-specific parser or suppressing protocol exceptions. handle_events is
synchronous, so an ASGI response task cannot run before cancellation below.
"""
from uvicorn.protocols.http.h11_impl import H11Protocol


class SDVH11Protocol(H11Protocol):
    def handle_events(self):
        super().handle_events()
        if self.cycle is not None and not self.cycle.response_complete:
            self._unset_keepalive_if_required()

    def handle_websocket_upgrade(self, event):
        # Uvicorn transfers the transport before creating an HTTP cycle.
        # Clear the old HTTP idle timer before that protocol hand-off too.
        self._unset_keepalive_if_required()
        super().handle_websocket_upgrade(event)
