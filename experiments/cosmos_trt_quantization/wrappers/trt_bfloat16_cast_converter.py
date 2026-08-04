"""Torch-TensorRT 2.7 compatibility converter for aten._to_copy to BF16.

The built-in converter implementation supports BF16, but its capability
validator omits ``torch.bfloat16`` from the allow-list.  Register the existing
implementation at high priority only for that exact dtype.
"""

from __future__ import annotations

import torch


def register_bfloat16_to_copy_converter() -> None:
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        ConverterPriority,
        dynamo_tensorrt_converter,
    )
    from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
        aten_ops_clone_copy_dtype,
    )

    def is_bfloat16_copy(node, _settings=None) -> bool:
        return node.kwargs.get("dtype") is torch.bfloat16

    dynamo_tensorrt_converter(
        torch.ops.aten._to_copy.default,
        capability_validator=is_bfloat16_copy,
        priority=ConverterPriority.HIGH,
        supports_dynamic_shapes=True,
    )(aten_ops_clone_copy_dtype)

