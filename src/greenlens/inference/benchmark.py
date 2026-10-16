import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from greenlens.models.architecture import build_resnet18


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "best_model.pth"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="GreenLens Model Latency & Architecture Benchmarking")
    parser.add_argument("--warmup-runs", type=int, default=20, help="Number of warmup iterations to exclude")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Number of timed benchmark iterations")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32], help="Batch sizes to evaluate")
    return parser.parse_args()


def measure_gpu_latency(model, batch_size, warmups=20, runs=100):
    device = torch.device("cuda")
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    model.eval()

    # 1. Warm-up phase: Run without timing to ensure CUDA context, memory caching, and kernels are initialized
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(dummy_input)
            torch.cuda.synchronize()

    # 2. Benchmark phase: Use torch.cuda.synchronize() for accurate GPU kernel execution timing
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    fps = float((batch_size / (mean_lat / 1000.0)))
    per_sample_lat = float(mean_lat / batch_size)

    return {
        "batch_size": batch_size,
        "mean_ms": round(mean_lat, 3),
        "std_ms": round(std_lat, 3),
        "median_p50_ms": round(p50_lat, 3),
        "p95_ms": round(p95_lat, 3),
        "p99_ms": round(p99_lat, 3),
        "per_sample_mean_ms": round(per_sample_lat, 3),
        "throughput_fps": round(fps, 1),
    }


def measure_cpu_latency(model, batch_size=1, warmups=10, runs=50):
    device = torch.device("cpu")
    cpu_model = model.to(device)
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    cpu_model.eval()

    with torch.no_grad():
        for _ in range(warmups):
            _ = cpu_model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = cpu_model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    fps = float((batch_size / (mean_lat / 1000.0)))

    return {
        "batch_size": batch_size,
        "mean_ms": round(mean_lat, 3),
        "median_p50_ms": round(p50_lat, 3),
        "p95_ms": round(p95_lat, 3),
        "throughput_fps": round(fps, 1),
    }


def main():
    args = parse_args()

    print("=" * 75)
    print("⏱️  GreenLens Model Latency & Architecture Benchmarking")
    print(f"Warmup runs: {args.warmup_runs} (excluded from stats)")
    print(f"Benchmark runs: {args.benchmark_runs} (using torch.cuda.synchronize())")
    print("=" * 75)

    # 1. Load Model Checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint.get("class_to_idx", {})
    num_classes = len(class_to_idx) if class_to_idx else 15

    model = build_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Compute Parameters & Storage
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

    print("\n📦 Model Architecture & Parameter Stats:")
    print(f"  • Architecture:          ResNet-18 (Pretrained backbone + Linear FC)")
    print(f"  • Number of classes:     {num_classes}")
    print(f"  • Total parameters:      {total_params:,} (~{total_params / 1e6:.2f} Million)")
    print(f"  • Trainable parameters:  {trainable_params:,}")
    print(f"  • Checkpoint File Size:  {model_file_size_mb:.2f} MB")
    print(f"  • Theoretical FLOPs:     ~1.82 GFLOPs (for 224x224 RGB input)")

    # 3. GPU Latency Benchmarking
    gpu_available = torch.cuda.is_available()
    gpu_results = []
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n⚡ GPU Latency Benchmarking on [{gpu_name}]:")
        model = model.to("cuda")

        print(f"{'Batch':^7} | {'Mean (ms)':^11} | {'Median (ms)':^12} | {'P95 (ms)':^10} | {'Per-Sample (ms)':^16} | {'Throughput (FPS)':^16}")
        print("-" * 85)

        for b in args.batch_sizes:
            res = measure_gpu_latency(model, batch_size=b, warmups=args.warmup_runs, runs=args.benchmark_runs)
            gpu_results.append(res)
            print(
                f"{res['batch_size']:^7} | {res['mean_ms']:^11.2f} | {res['median_p50_ms']:^12.2f} | {res['p95_ms']:^10.2f} | {res['per_sample_mean_ms']:^16.2f} | {res['throughput_fps']:^16.1f}"
            )

    # 4. CPU Latency Benchmarking (for edge/serverless comparison)
    print("\n💻 CPU Latency Benchmarking (Batch Size = 1 for edge/server inference):")
    cpu_res = measure_cpu_latency(model, batch_size=1, warmups=10, runs=30)
    print(f"  • CPU Mean Latency:   {cpu_res['mean_ms']:.2f} ms")
    print(f"  • CPU Median (P50):   {cpu_res['median_p50_ms']:.2f} ms")
    print(f"  • CPU 95th percentile:{cpu_res['p95_ms']:.2f} ms")
    print(f"  • CPU Throughput:     {cpu_res['throughput_fps']:.1f} FPS")

    # 5. Export JSON Report
    report = {
        "model_architecture": "ResNet-18",
        "num_classes": num_classes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_file_size_mb": round(model_file_size_mb, 2),
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "gpu_benchmarks": gpu_results,
        "cpu_benchmark": cpu_res,
        "best_epoch": checkpoint.get("epoch"),
        "best_macro_f1": checkpoint.get("macro_f1"),
        "best_balanced_acc": checkpoint.get("balanced_acc"),
        "best_val_acc": checkpoint.get("val_acc"),
        "best_val_loss": checkpoint.get("val_loss"),
    }

    report_path = REPORTS_DIR / "model_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n📄 Benchmark report exported to: {report_path}")


if __name__ == "__main__":
    main()
