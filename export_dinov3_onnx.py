#!/usr/bin/env python3
"""Export Meta DINOv3 ViT-S/16 to ONNX and validate with ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


# Configuration
REPOSITORY = Path("/home/devuser/WASP/dinov3")
CHECKPOINT = Path(
    "/home/devuser/WASP/dinov3/pretrained/dinov3_vits16.pth"
)
OUTPUT = Path(
 #   "/home/devuser/WASP/dinov3/pretrained/dinov3_vits16.onnx"
    "/home/devuser/WASP/dinov3/pretrained/dinov3_vits16_inline.onnx"
)

MODEL_NAME = "dinov3_vits16"
IMAGE_SIZE = 224
OPSET_VERSION = 18


class DINOv3ONNX(torch.nn.Module):
    """Expose CLS and patch embeddings as ONNX outputs."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        features = self.backbone.forward_features(images)

        return (
            features["x_norm_clstoken"],
            features["x_norm_patchtokens"],
        )


def main():
    if not REPOSITORY.is_dir():
        raise FileNotFoundError(REPOSITORY)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {REPOSITORY}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Model:      {MODEL_NAME}")
    print(f"Output:     {OUTPUT}")

    # Load pretrained model from local repository and checkpoint.
    backbone = torch.hub.load(
        str(REPOSITORY),
        MODEL_NAME,
        source="local",
        weights=str(CHECKPOINT),
    )

    backbone = backbone.cpu().eval()
    model = DINOv3ONNX(backbone).eval()

    sample = torch.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE,
        dtype=torch.float32,
    )

    # Export to ONNX
    print("Exporting ONNX...")

    torch.onnx.export(
        model,
        (sample,),
        str(OUTPUT),
        input_names=["images"],
        output_names=[
            "cls_embedding",
            "patch_embeddings",
        ],
        opset_version=OPSET_VERSION,
        dynamo=True,
        external_data=False,
    )

    # Validate ONNX graph
    print("Checking ONNX model...")
    onnx.checker.check_model(str(OUTPUT))

    # Validate outputs against PyTorch
    print("Comparing ONNX Runtime with PyTorch...")

    session = ort.InferenceSession(
        str(OUTPUT),
        providers=["CPUExecutionProvider"],
    )

    with torch.inference_mode():
        pt_outputs = model(sample)

    ort_outputs = session.run(
        None,
        {"images": sample.numpy()},
    )

    for name, pt, ort_result in zip(
        ["cls_embedding", "patch_embeddings"],
        pt_outputs,
        ort_outputs,
    ):
        reference = pt.detach().numpy()

        max_error = np.max(
            np.abs(reference - ort_result)
        )

        print(
            f"{name}: shape={ort_result.shape}, "
            f"max_error={max_error:.8f}"
        )

        np.testing.assert_allclose(
            reference,
            ort_result,
            rtol=1e-3,
            atol=1e-4,
        )

    print(f"ONNX export successful: {OUTPUT}")


if __name__ == "__main__":
    main()

