"""Minimal hand-written gRPC client stub for the entropy service.

Provides ``EntropyServiceStub`` which is the only class the qc-sampler
client code needs. The server-side servicer and add-to-server helpers
are omitted because the QRNG server is a separate codebase.
"""

from __future__ import annotations

import grpc

from qc_sampler.proto import entropy_service_pb2 as entropy__service__pb2


class EntropyServiceStub:
    """Client stub for the EntropyService gRPC service.

    Provides two RPC methods:
    - GetEntropy: unary-unary call to fetch a batch of random bytes.
    - StreamEntropy: bidirectional streaming (reserved for future use).
    """

    def __init__(self, channel: grpc.Channel) -> None:
        """Initialize the stub with a gRPC channel.

        Args:
            channel: An open gRPC channel to the entropy service.
        """
        self.GetEntropy = channel.unary_unary(
            "/entropy.EntropyService/GetEntropy",
            request_serializer=entropy__service__pb2.EntropyRequest.SerializeToString,
            response_deserializer=entropy__service__pb2.EntropyResponse.FromString,
        )
        self.StreamEntropy = channel.stream_stream(
            "/entropy.EntropyService/StreamEntropy",
            request_serializer=entropy__service__pb2.EntropyRequest.SerializeToString,
            response_deserializer=entropy__service__pb2.EntropyResponse.FromString,
        )
