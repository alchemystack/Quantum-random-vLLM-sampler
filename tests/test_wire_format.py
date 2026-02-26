"""Tests for protobuf wire-format encoding in hand-written proto stubs.

Validates that ``EntropyRequest`` and ``EntropyResponse`` produce standard
protobuf wire-format bytes, enabling interoperability with any standard
gRPC server (including ``grpcurl`` and the user's ``qrng.QuantumRNG``).
"""

from __future__ import annotations

import pytest

from qr_sampler.proto.entropy_service_pb2 import (
    EntropyRequest,
    EntropyResponse,
    _decode_varint,
    _encode_varint,
)

# ---------------------------------------------------------------------------
# Varint encoding/decoding
# ---------------------------------------------------------------------------


class TestVarint:
    """Test low-level varint encoding and decoding."""

    @pytest.mark.parametrize(
        ("value", "expected_bytes"),
        [
            (0, b"\x00"),
            (1, b"\x01"),
            (127, b"\x7f"),
            (128, b"\x80\x01"),
            (300, b"\xac\x02"),
            (16384, b"\x80\x80\x01"),
            (20480, b"\x80\xa0\x01"),
        ],
    )
    def test_encode_known_values(self, value: int, expected_bytes: bytes) -> None:
        assert _encode_varint(value) == expected_bytes

    @pytest.mark.parametrize(
        ("encoded", "expected_value"),
        [
            (b"\x00", 0),
            (b"\x01", 1),
            (b"\x7f", 127),
            (b"\x80\x01", 128),
            (b"\xac\x02", 300),
            (b"\x80\xa0\x01", 20480),
        ],
    )
    def test_decode_known_values(self, encoded: bytes, expected_value: int) -> None:
        value, offset = _decode_varint(encoded, 0)
        assert value == expected_value
        assert offset == len(encoded)

    def test_roundtrip(self) -> None:
        for v in [0, 1, 42, 127, 128, 255, 256, 1000, 20480, 65535, 2**20]:
            encoded = _encode_varint(v)
            decoded, _ = _decode_varint(encoded, 0)
            assert decoded == v


# ---------------------------------------------------------------------------
# EntropyRequest wire format
# ---------------------------------------------------------------------------


class TestEntropyRequestWireFormat:
    """Test that EntropyRequest produces standard protobuf encoding."""

    def test_empty_request_serializes_to_empty(self) -> None:
        """Proto3: all-default-valued fields produce empty bytes."""
        req = EntropyRequest(bytes_needed=0, sequence_id=0)
        assert req.SerializeToString() == b""

    def test_known_encoding_bytes_needed_only(self) -> None:
        """Field 1, varint 100 -> tag=0x08, value=0x64."""
        req = EntropyRequest(bytes_needed=100)
        wire = req.SerializeToString()
        # Tag: field_number=1, wire_type=0 -> (1<<3)|0 = 0x08
        # Value: 100 = 0x64
        assert wire == b"\x08\x64"

    def test_known_encoding_both_fields(self) -> None:
        """bytes_needed=100, sequence_id=42."""
        req = EntropyRequest(bytes_needed=100, sequence_id=42)
        wire = req.SerializeToString()
        # Field 1: tag=0x08, value=100 (0x64)
        # Field 2: tag=0x10, value=42 (0x2a)
        assert wire == b"\x08\x64\x10\x2a"

    def test_known_encoding_large_varint(self) -> None:
        """bytes_needed=20480 requires multi-byte varint."""
        req = EntropyRequest(bytes_needed=20480)
        wire = req.SerializeToString()
        # Tag: 0x08, Value: 20480 = 0x80 0xa0 0x01
        assert wire == b"\x08\x80\xa0\x01"

    def test_roundtrip(self) -> None:
        req = EntropyRequest(bytes_needed=20480, sequence_id=99)
        wire = req.SerializeToString()
        decoded = EntropyRequest.FromString(wire)
        assert decoded.bytes_needed == 20480
        assert decoded.sequence_id == 99

    def test_roundtrip_defaults(self) -> None:
        req = EntropyRequest()
        wire = req.SerializeToString()
        decoded = EntropyRequest.FromString(wire)
        assert decoded.bytes_needed == 0
        assert decoded.sequence_id == 0

    def test_from_string_skips_unknown_fields(self) -> None:
        """Unknown fields should be silently skipped."""
        # Valid EntropyRequest(bytes_needed=100) + an unknown field 5 varint=99
        wire = b"\x08\x64" + b"\x28\x63"
        decoded = EntropyRequest.FromString(wire)
        assert decoded.bytes_needed == 100
        assert decoded.sequence_id == 0

    def test_from_string_empty_bytes(self) -> None:
        decoded = EntropyRequest.FromString(b"")
        assert decoded.bytes_needed == 0
        assert decoded.sequence_id == 0


# ---------------------------------------------------------------------------
# EntropyResponse wire format
# ---------------------------------------------------------------------------


class TestEntropyResponseWireFormat:
    """Test that EntropyResponse produces standard protobuf encoding."""

    def test_empty_response_serializes_to_empty(self) -> None:
        resp = EntropyResponse()
        assert resp.SerializeToString() == b""

    def test_known_encoding_data_only(self) -> None:
        """Field 1 (bytes), 3 bytes of data."""
        resp = EntropyResponse(data=b"\xaa\xbb\xcc")
        wire = resp.SerializeToString()
        # Tag: field_number=1, wire_type=2 -> (1<<3)|2 = 0x0a
        # Length: 3 = 0x03
        # Data: \xaa\xbb\xcc
        assert wire == b"\x0a\x03\xaa\xbb\xcc"

    def test_known_encoding_all_fields(self) -> None:
        """All four fields populated."""
        resp = EntropyResponse(
            data=b"\x42",
            sequence_id=1,
            generation_timestamp_ns=1000,
            device_id="dev1",
        )
        wire = resp.SerializeToString()
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == b"\x42"
        assert decoded.sequence_id == 1
        assert decoded.generation_timestamp_ns == 1000
        assert decoded.device_id == "dev1"

    def test_roundtrip(self) -> None:
        resp = EntropyResponse(
            data=b"\x00\x01\x02\x03\x04" * 100,
            sequence_id=12345,
            generation_timestamp_ns=1_700_000_000_000_000_000,
            device_id="firefly-1",
        )
        wire = resp.SerializeToString()
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == resp.data
        assert decoded.sequence_id == 12345
        assert decoded.generation_timestamp_ns == 1_700_000_000_000_000_000
        assert decoded.device_id == "firefly-1"

    def test_roundtrip_defaults(self) -> None:
        resp = EntropyResponse()
        wire = resp.SerializeToString()
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == b""
        assert decoded.sequence_id == 0
        assert decoded.generation_timestamp_ns == 0
        assert decoded.device_id == ""

    def test_from_string_skips_unknown_fields(self) -> None:
        """Unknown fields should be silently skipped."""
        resp = EntropyResponse(data=b"\x42", sequence_id=1)
        wire = resp.SerializeToString()
        # Append an unknown field 10, varint=99
        wire += b"\x50\x63"
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == b"\x42"
        assert decoded.sequence_id == 1

    def test_from_string_empty_bytes(self) -> None:
        decoded = EntropyResponse.FromString(b"")
        assert decoded.data == b""
        assert decoded.sequence_id == 0

    def test_large_data_payload(self) -> None:
        """Test with the typical 20KB entropy payload."""
        payload = bytes(range(256)) * 80  # 20480 bytes
        resp = EntropyResponse(data=payload, device_id="test")
        wire = resp.SerializeToString()
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == payload
        assert len(decoded.data) == 20480


# ---------------------------------------------------------------------------
# Cross-compatibility with generic wire-format decoder
# ---------------------------------------------------------------------------


class TestCrossCompatibility:
    """Verify that EntropyRequest wire output is decodable by a generic
    protobuf field-1 extractor (the pattern used in quantum.py for
    protocol-agnostic gRPC).
    """

    def test_request_field1_is_varint(self) -> None:
        """EntropyRequest.SerializeToString() field 1 should be decodable
        as a generic varint.
        """
        req = EntropyRequest(bytes_needed=20480)
        wire = req.SerializeToString()
        # Parse the tag
        tag, offset = _decode_varint(wire, 0)
        field_number = tag >> 3
        wire_type = tag & 0x07
        assert field_number == 1
        assert wire_type == 0  # varint
        value, _ = _decode_varint(wire, offset)
        assert value == 20480

    def test_response_field1_is_length_delimited(self) -> None:
        """EntropyResponse.SerializeToString() field 1 should be decodable
        as a generic length-delimited bytes extraction.
        """
        payload = b"\xde\xad\xbe\xef"
        resp = EntropyResponse(data=payload, sequence_id=1)
        wire = resp.SerializeToString()
        # Parse the tag
        tag, offset = _decode_varint(wire, 0)
        field_number = tag >> 3
        wire_type = tag & 0x07
        assert field_number == 1
        assert wire_type == 2  # length-delimited
        length, offset = _decode_varint(wire, offset)
        assert length == 4
        assert wire[offset : offset + length] == payload

    def test_request_with_only_field1_produces_minimal_wire(self) -> None:
        """A request with only bytes_needed set should produce the same wire
        bytes regardless of whether it's an EntropyRequest or a generic
        'field 1 varint' encoder.
        """
        req = EntropyRequest(bytes_needed=100)
        wire = req.SerializeToString()
        # Manually construct: tag(1, varint) + varint(100)
        expected = _encode_varint((1 << 3) | 0) + _encode_varint(100)
        assert wire == expected
