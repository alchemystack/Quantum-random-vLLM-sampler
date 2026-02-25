"""Hand-written protobuf message stubs for the entropy service.

These are lightweight message classes that mirror the ``entropy_service.proto``
definition without requiring ``protoc`` code generation. They provide attribute
access compatible with the gRPC stub expectations.

If the proto definition changes, update these stubs or regenerate with
``grpc_tools.protoc``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntropyRequest:
    """Entropy generation request message.

    Attributes:
        bytes_needed: Number of random bytes requested.
        sequence_id: Correlates request to response in streaming mode.
    """

    bytes_needed: int = 0
    sequence_id: int = 0

    def SerializeToString(self) -> bytes:  # noqa: N802
        """Serialize to a simple wire format.

        Uses a minimal custom encoding rather than full protobuf serialization
        to avoid a hard dependency on the protobuf C extension for the message
        classes themselves.
        """
        import struct

        return struct.pack(">iQ", self.bytes_needed, self.sequence_id)

    @classmethod
    def FromString(cls, data: bytes) -> EntropyRequest:  # noqa: N802
        """Deserialize from wire format."""
        import struct

        bytes_needed, sequence_id = struct.unpack(">iQ", data)
        return cls(bytes_needed=bytes_needed, sequence_id=sequence_id)


@dataclass
class EntropyResponse:
    """Entropy generation response message.

    Attributes:
        data: Random bytes.
        sequence_id: Matches the request's ``sequence_id``.
        generation_timestamp_ns: When physical entropy was generated (nanoseconds).
        device_id: QRNG hardware identifier.
    """

    data: bytes = field(default_factory=bytes)
    sequence_id: int = 0
    generation_timestamp_ns: int = 0
    device_id: str = ""

    def SerializeToString(self) -> bytes:  # noqa: N802
        """Serialize to a simple wire format."""
        import struct

        device_bytes = self.device_id.encode("utf-8")
        return struct.pack(
            f">I{len(self.data)}sQQ H{len(device_bytes)}s",
            len(self.data),
            self.data,
            self.sequence_id,
            self.generation_timestamp_ns,
            len(device_bytes),
            device_bytes,
        )

    @classmethod
    def FromString(cls, data: bytes) -> EntropyResponse:  # noqa: N802
        """Deserialize from wire format."""
        import struct

        offset = 0
        (data_len,) = struct.unpack_from(">I", data, offset)
        offset += 4
        entropy_data = data[offset : offset + data_len]
        offset += data_len
        sequence_id, gen_ts = struct.unpack_from(">QQ", data, offset)
        offset += 16
        (dev_len,) = struct.unpack_from(">H", data, offset)
        offset += 2
        device_id = data[offset : offset + dev_len].decode("utf-8")
        return cls(
            data=entropy_data,
            sequence_id=sequence_id,
            generation_timestamp_ns=gen_ts,
            device_id=device_id,
        )
