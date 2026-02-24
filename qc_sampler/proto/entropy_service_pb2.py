"""Minimal hand-written protobuf message stubs for the entropy service.

These classes mirror what ``protoc`` would generate from
``entropy_service.proto`` but avoid the build-time compilation step.
Only the fields used by the gRPC client (EntropyRequest, EntropyResponse)
are implemented.

The descriptor is built programmatically from a FileDescriptorProto to
avoid version-specific serialized byte strings.  This makes the stubs
compatible across protobuf runtime versions (v4, v5, v6+).
"""

from __future__ import annotations

from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder


def _build_file_descriptor_proto() -> bytes:
    """Construct the serialized FileDescriptorProto at import time.

    Building the proto descriptor programmatically (instead of embedding
    a pre-serialized byte string) guarantees compatibility across
    protobuf runtime versions.
    """
    fdp = _descriptor_pb2.FileDescriptorProto()
    fdp.name = "entropy_service.proto"
    fdp.package = "entropy"
    fdp.syntax = "proto3"

    # EntropyRequest { int32 bytes_needed = 1; }
    msg_req = fdp.message_type.add()
    msg_req.name = "EntropyRequest"
    f = msg_req.field.add()
    f.name = "bytes_needed"
    f.number = 1
    f.type = _descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    f.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    # EntropyResponse { bytes data = 1; int64 timestamp_ns = 2; string device_id = 3; }
    msg_resp = fdp.message_type.add()
    msg_resp.name = "EntropyResponse"

    f = msg_resp.field.add()
    f.name = "data"
    f.number = 1
    f.type = _descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    f.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    f = msg_resp.field.add()
    f.name = "timestamp_ns"
    f.number = 2
    f.type = _descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    f.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    f = msg_resp.field.add()
    f.name = "device_id"
    f.number = 3
    f.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    f.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    return fdp.SerializeToString()


_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    _build_file_descriptor_proto()
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "entropy_service_pb2", _globals
)

EntropyRequest = _globals["EntropyRequest"]
EntropyResponse = _globals["EntropyResponse"]
