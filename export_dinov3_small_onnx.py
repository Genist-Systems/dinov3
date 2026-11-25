from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np

ROOT = Path("/home/devuser/WASP/dinov3")
CHECKPOINT = ROOT / "pretrained/dinov3_vits16.pth"

BLOCKS = 4
OUTPUT = ROOT / f"pretrained/dinov3_vits16_{BLOCKS}blocks.onnx"

backbone = torch.hub.load(
    str(ROOT),
    "dinov3_vits16",
    source="local",
    weights=str(CHECKPOINT),
)

backbone.eval().cpu()

original_blocks = len(backbone.blocks)
print("Original transformer blocks:", original_blocks)

# Retain only the first N pretrained transformer blocks
backbone.blocks = torch.nn.ModuleList(
    list(backbone.blocks)[:BLOCKS]
)

print("Exporting blocks:", len(backbone.blocks))


class ExportModel(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        features = self.backbone.forward_features(images)
        return (
            features["x_norm_clstoken"],
            features["x_norm_patchtokens"],
        )


model = ExportModel(backbone).eval()
sample = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    (sample,),
    str(OUTPUT),
    input_names=["images"],
    output_names=["cls_embedding", "patch_embeddings"],
    opset_version=18,
    dynamo=True,
    external_data=False,
)

onnx.checker.check_model(str(OUTPUT))

graph = onnx.load(str(OUTPUT), load_external_data=False)

elements = sum(
    int(np.prod(t.dims))
    for t in graph.graph.initializer
)

assert all(
    t.data_location != onnx.TensorProto.EXTERNAL
    for t in graph.graph.initializer
)

session = ort.InferenceSession(
    str(OUTPUT),
    providers=["CPUExecutionProvider"],
)

with torch.inference_mode():
    reference = model(sample)

results = session.run(None, {"images": sample.numpy()})

for expected, actual in zip(reference, results):
    np.testing.assert_allclose(
        expected.numpy(),
        actual,
        rtol=1e-3,
        atol=1e-4,
    )

print("Output:", OUTPUT)
print("Size MiB:", OUTPUT.stat().st_size / 1048576)
print("Weight elements:", f"{elements:,}")
print("ONNX validation: PASSED")

