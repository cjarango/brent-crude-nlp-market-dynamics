"""
build_onnx
==========

Exports trained PyTorch models (LSTM or GRU) to ONNX format and runs a strict round-trip test.
Updates metadata.json with the round-trip validation results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from src.models.gru_wrapper import GRUNetwork
from src.models.lstm_wrapper import LSTMNetwork


def export_and_verify(artifact_dir: Path, model_type: str):
    """Exports a PyTorch model to ONNX and verifies numerical equivalence.

    Args:
        artifact_dir: Directory containing model_state_dict.pt and metadata.json.
        model_type: Either 'lstm' or 'gru'.
    """
    meta_path = artifact_dir / "metadata.json"
    weights_path = artifact_dir / "model_state_dict.pt"
    fixed_input_path = artifact_dir / "roundtrip_test_input.npy"
    
    if not meta_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing artifacts in {artifact_dir}")

    print(f"[{model_type.upper()}] Loading metadata from {meta_path}")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    params = metadata["params"]
    context_len = metadata["context_length"]

    if model_type == "lstm":
        net = LSTMNetwork(
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params.get("dropout", 0.0)
        )
    elif model_type == "gru":
        net = GRUNetwork(
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params.get("dropout", 0.0)
        )
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")

    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.eval()

    dummy_input = torch.randn(1, context_len, 1, dtype=torch.float32)
    onnx_path = artifact_dir / "model.onnx"
    
    print(f"[{model_type.upper()}] Exporting to {onnx_path}")
    torch.onnx.export(
        net,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["variance"],
        dynamic_axes={"input": {0: "batch_size"}, "variance": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"[{model_type.upper()}] ONNX graph structure is valid.")

    print(f"[{model_type.upper()}] Running Round-Trip Test")
    
    if fixed_input_path.exists():
        test_inputs = np.load(fixed_input_path)
        test_inputs_t = torch.tensor(test_inputs, dtype=torch.float32)
    else:
        torch.manual_seed(42)
        test_inputs_t = torch.randn(32, context_len, 1, dtype=torch.float32)
        test_inputs = test_inputs_t.numpy()
        np.save(fixed_input_path, test_inputs)
        print(f"[{model_type.upper()}] Created fixed test batch at {fixed_input_path}")

    with torch.no_grad():
        torch_out = net(test_inputs_t).numpy()
        
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": test_inputs})[0]
    
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    tolerance = 1e-4
    
    print(f"  - Max absolute difference: {max_diff:.2e}")
    if max_diff > tolerance:
        raise RuntimeError(f"Round-trip FAILED: {max_diff} > {tolerance}")
    print("  - Round-trip PASSED")

    metadata["roundtrip_test"] = {
        "max_abs_error": max_diff,
        "tolerance": tolerance,
        "passed": True,
        "opset_version": 17,
        "test_batch_shape": list(test_inputs.shape),
        "onnx_provider": "CPUExecutionProvider"
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[{model_type.upper()}] Metadata updated at {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch RNN to ONNX")
    parser.add_argument("--dir", type=str, required=True, help="Path to artifact directory (e.g., models/lstm_final)")
    parser.add_argument("--type", type=str, choices=["lstm", "gru"], required=True, help="Model type")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    export_and_verify(target_dir, args.type)