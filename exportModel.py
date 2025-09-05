import torch
import argparse
import os
import sys
import time
from datetime import datetime
import yaml
from ultralytics import YOLO


def is_multiprocessing_worker():
    """Check if this is a multiprocessing worker process"""
    return '--multiprocessing-fork' in sys.argv or 'parent_pid=' in ' '.join(sys.argv)


def parse_imgsz(value: str):
    """
    Parse --imgsz argument.
    Examples:
      --imgsz 640           -> 640
      --imgsz [1280, 720]   -> [1280, 720]
    """
    value = value.strip()

    # Square case
    if value.isdigit():
        return int(value)

    # List-like case: [h, w]
    if value.startswith("[") and value.endswith("]"):
        try:
            parts = value[1:-1].split(",")
            if len(parts) != 2:
                raise ValueError
            h, w = [int(p.strip()) for p in parts]
            return (h, w)
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Invalid --imgsz format '{value}'. Use int or [h, w]."
            )

    raise argparse.ArgumentTypeError(
        f"Invalid --imgsz format '{value}'. Use int or [h, w]."
    )


def parse_args():
    if is_multiprocessing_worker():
        # print("Detected multiprocessing worker - skipping argument parsing")
        return None
    p = argparse.ArgumentParser(description="Convert Ultralytics .pt to another format (TorchScript/ONNX/engine/etc.)")
    p.add_argument("--model", required=True, help="Path to .pt weights (Ultralytics checkpoint)")
    p.add_argument("--format", default="torchscript",
                   help="Target export format: 'torchscript', 'onnx', 'engine' (TensorRT), 'openvino', etc.")
    p.add_argument("--imgsz", type=parse_imgsz, default="640",
                   help="Image size: int (e.g. 640) for square, or [720,1280].")
    p.add_argument("--keras", action="store_true", help="Export to Keras (TensorFlow SavedModel) if supported.")
    p.add_argument("--optimize", action="store_true", help="Apply TorchScript/mobile optimization.")
    p.add_argument("--half", action="store_true", help="Export FP16 where supported.")
    p.add_argument("--int8", action="store_true", help="Enable INT8 quantization (PTQ) where supported.")
    p.add_argument("--dynamic", action="store_true",
                   help="Allow dynamic input sizes (ONNX/OpenVINO/TensorRT where supported).")
    p.add_argument("--simplify", action="store_true", default=True,
                   help="Simplify ONNX model using onnxslim (default True).")
    p.add_argument("--no-simplify", dest="simplify", action="store_false", help="Disable ONNX simplification.")
    p.add_argument("--opset", type=int, default=None, help="ONNX opset version (int).")
    p.add_argument("--workspace", type=float, default=None, help="TensorRT workspace size in GiB (float).")
    p.add_argument("--nms", action="store_true", help="Add NMS into the exported model when supported.")
    p.add_argument("--batch", type=int, default=1, help="Batch size for export / deployed model.")
    p.add_argument("--device", type=str, default=None,
                   help="Device to export on: 'cpu', '0' (gpu0), 'mps', 'dla:0', etc. (TensorRT will force GPU).")
    p.add_argument("--data", type=str, default="coco8.yaml",
                   help="Path to dataset config used for INT8 calibration (default coco8.yaml).")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of dataset to use for INT8 calibration (0.0-1.0).")
    p.add_argument("-o", "--output", type=str, default=None, help="Explicit output filename/path for exported model.")
    p.add_argument("--verbose", action="store_true", help="Print verbose debug info.")
    return p.parse_args()


def main():
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("CUDA version:", torch.version.cuda, flush=True)

    try:
        args = parse_args()
        if args is None:
            sys.exit(0)
        print(f"Checking model file: {args.model}", flush=True)
        if not os.path.exists(args.model):
            print(f"ERROR: Model file not found: {args.model}", flush=True)
            print(f"Full path: {os.path.abspath(args.model)}", flush=True)
            sys.exit(1)
        else:
            print(f"Model file exists: {args.model}", flush=True)
            print(f"File size: {os.path.getsize(args.model)} bytes", flush=True)
        export_kwargs = {
            "model": args.model,
            "format": args.format,
            "imgsz": args.imgsz,
            "keras": args.keras,
            "optimize": args.optimize,
            "dynamic": args.dynamic,
            "batch": args.batch,
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "data": args.data,
            "fraction": args.fraction,
            "simplify": args.simplify,
            "opset": args.opset,
            "workspace": args.workspace,
            "nms": args.nms,
        }
        try:
            model = YOLO(args.model)
            results = model.export(**export_kwargs)
            print(f"Export successful: {results}", flush=True)
            sys.exit(0)
        except Exception as e:
            print(f"ERROR during exporting: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        print(f"ERROR in main: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        import multiprocessing
    except Exception:
        pass

    if hasattr(sys, "frozen"):
        try:
            multiprocessing.freeze_support()
        except Exception:
            pass
    main()
