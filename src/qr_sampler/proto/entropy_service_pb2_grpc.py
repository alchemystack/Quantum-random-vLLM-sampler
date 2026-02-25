"""Hand-written gRPC client stubs for the entropy service.

Provides ``EntropyServiceStub`` with both unary (``GetEntropy``) and
bidirectional streaming (``StreamEntropy``) RPCs. These stubs are
compatible with both sync and async (``grpc.aio``) channels.

If the proto definition changes, update these stubs or regenerate with
``grpc_tools.protoc``.
"""

from __future__ import annotations

from typing import Any


def _entropy_request_serializer(request: Any) -> bytes:
    """Serialize an EntropyRequest to bytes."""
    result: bytes = request.SerializeToString()
    return result


def _entropy_response_deserializer(data: bytes) -> Any:
    """Deserialize bytes to an EntropyResponse."""
    from qr_sampler.proto.entropy_service_pb2 import EntropyResponse

    return EntropyResponse.FromString(data)


def _entropy_request_deserializer(data: bytes) -> Any:
    """Deserialize bytes to an EntropyRequest."""
    from qr_sampler.proto.entropy_service_pb2 import EntropyRequest

    return EntropyRequest.FromString(data)


def _entropy_response_serializer(response: Any) -> bytes:
    """Serialize an EntropyResponse to bytes."""
    result: bytes = response.SerializeToString()
    return result


class EntropyServiceStub:
    """gRPC client stub for the EntropyService.

    Supports both sync and async (``grpc.aio``) channels. Provides two
    RPC methods:

    - ``GetEntropy``: unary request-response
    - ``StreamEntropy``: bidirectional streaming

    Args:
        channel: A gRPC Channel or async Channel instance.
    """

    def __init__(self, channel: Any) -> None:
        self.GetEntropy = channel.unary_unary(
            "/qr_entropy.EntropyService/GetEntropy",
            request_serializer=_entropy_request_serializer,
            response_deserializer=_entropy_response_deserializer,
        )
        self.StreamEntropy = channel.stream_stream(
            "/qr_entropy.EntropyService/StreamEntropy",
            request_serializer=_entropy_request_serializer,
            response_deserializer=_entropy_response_deserializer,
        )


class EntropyServiceServicer:
    """Base class for EntropyService server implementations.

    Override the methods in this class to implement the service.
    Used by example servers.
    """

    def GetEntropy(  # noqa: N802
        self,
        request: Any,
        context: Any,
    ) -> Any:
        """Unary RPC: single request -> single response."""
        import grpc

        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def StreamEntropy(  # noqa: N802
        self,
        request_iterator: Any,
        context: Any,
    ) -> Any:
        """Bidirectional streaming RPC."""
        import grpc

        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_EntropyServiceServicer_to_server(  # noqa: N802
    servicer: EntropyServiceServicer,
    server: Any,
) -> None:
    """Register an ``EntropyServiceServicer`` with a gRPC server.

    Args:
        servicer: The service implementation.
        server: A ``grpc.Server`` or ``grpc.aio.Server`` instance.
    """
    from grpc import stream_stream_rpc_method_handler, unary_unary_rpc_method_handler

    rpc_method_handlers = {
        "GetEntropy": unary_unary_rpc_method_handler(
            servicer.GetEntropy,
            request_deserializer=_entropy_request_deserializer,
            response_serializer=_entropy_response_serializer,
        ),
        "StreamEntropy": stream_stream_rpc_method_handler(
            servicer.StreamEntropy,
            request_deserializer=_entropy_request_deserializer,
            response_serializer=_entropy_response_serializer,
        ),
    }
    from grpc import method_handlers_generic_handler

    generic_handler = method_handlers_generic_handler(
        "qr_entropy.EntropyService",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))
