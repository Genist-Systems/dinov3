#!/usr/bin/env python3
"""Store ONNX weights in FP16, with FP32 computation."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

ROOT = Path("/home/devuser/WASP/dinov3/pretrained")

SOURCE = ROOT / "dinov3_vits16_inline.onnx"
OUTPUT = ROOT / "dinov3_vits16_compact.onnx"

LIMIT = 64 * 1024 * 1024

print("Loading:", SOURCE)
model = onnx.load(str(SOURCE), load_external_data=True)

new_initializers = []
cast_nodes = []
converted = 0

for tensor in model.graph.initializer:
    if tensor.data_type != TensorProto.FLOAT:
        new_initializers.append(tensor)
        continue

    values = numpy_helper.to_array(tensor)

    # Convert stored weights from 32-bit to 16-bit.
    half_values = values.astype(np.float16)

    # Retain FP32 for tensors that cannot be safely
    # represented as FP16.
    if not np.all(np.isfinite(half_values)) and np.all(
        np.isfinite(values)
    ):
        new_initializers.append(tensor)
        continue

    stored_name = tensor.name + "__fp16_storage"

    new_initializers.append(
        numpy_helper.from_array(
            half_values,
            name=stored_name,
        )
    )

    # Restore FP32 before the original computation.
    cast_nodes.append(
        helper.make_node(
            "Cast",
            inputs=[stored_name],
            outputs=[tensor.name],
            to=TensorProto.FLOAT,
            name=f"RestoreFP32_{converted}",
        )
    )

    converted += 1

# Replace initializers
model.graph.ClearField("initializer")
model.graph.initializer.extend(new_initializers)

# Insert Cast operations before original graph nodes
original_nodes = list(model.graph.node)
model.graph.ClearField("node")
model.graph.node.extend(cast_nodes)
model.graph.node.extend(original_nodes)

print("Converted tensors:", converted)

# Save all weights inside the ONNX file
onnx.save_model(
    model,
    str(OUTPUT),
    save_as_external_data=False,
)

size = OUTPUT.stat().st_size

print("Output:", OUTPUT)
print(f"Size: {size / 1048576:.2f} MiB")

if size >= LIMIT:
    raise RuntimeError("Model still exceeds 64 MiB")

# Validate graph
onnx.checker.check_model(str(OUTPUT))

# Validate no external weights
check = onnx.load(str(OUTPUT), load_external_data=False)

assert all(
    t.data_location != TensorProto.EXTERNAL
    for t in check.graph.initializer
)

# Compare original and compact ONNX outputs
sample = np.random.randn(1, 3, 224, 224).astype(np.float32)

original_session = ort.InferenceSession(
    str(SOURCE),
    providers=["CPUExecutionProvider"],
)

compact_session = ort.InferenceSession(
    str(OUTPUT),
    providers=["CPUExecutionProvider"],
)

reference = original_session.run(None, {"images": sample})
candidate = compact_session.run(None, {"images": sample})

for name, a, b in zip(
    ["cls_embedding", "patch_embeddings"],
    reference,
    candidate,
):
    diff = np.abs(a - b)
    similarity = float(
        np.dot(a.ravel(), b.ravel())
        / (
            np.linalg.norm(a.ravel())
            * np.linalg.norm(b.ravel())
        )
    )

    print(
        f"{name}: "
        f"max_error={diff.max():.6f}, "
        f"mean_error={diff.mean():.6f}, "
        f"cosine={similarity:.8f}"
    )

print("SUCCESS: Compact ONNX passed structural and runtime checks")

