#!/usr/bin/env python3
"""Benchmark inference latency for CNN-LSTM and GNN-LSTM models.

This script profiles model forward passes on different hardware profiles
and compares performance characteristics for edge deployment.
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np


# Simple CNN-LSTM model architecture (matches your training)
class CNNLSTM(nn.Module):
    def __init__(self, past_len=10, future_len=20, cnn_channels=64, lstm_hidden=128):
        super().__init__()
        self.future_len = future_len
        self.cnn = nn.Sequential(
            nn.Conv1d(2, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, future_len * 2),
        )

    def forward(self, past_xy):
        x = past_xy.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        out = self.head(h[-1])
        return out.view(-1, self.future_len, 2)


def profile_model(model, dummy_input, device, num_warmup=10, num_iterations=100):
    """Profile inference latency of a model."""
    model = model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input.to(device))
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            _ = model(dummy_input.to(device))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }


def benchmark_all():
    """Run comprehensive benchmarks on available devices."""
    results = {}
    
    # Test data: batch_size=1 (edge device constraint), past_len=10
    dummy_input = torch.randn(1, 10, 2)
    
    models_to_test = [
        ("CNN-LSTM", CNNLSTM(past_len=10, future_len=20, cnn_channels=64, lstm_hidden=128)),
    ]
    
    # Test on available devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    
    print("=" * 70)
    print("NEURAL NETWORK INFERENCE LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Batch size: 1 (edge device)")
    print(f"Iterations: 100 (with 10 warmup)")
    print("=" * 70)
    
    for model_name, model in models_to_test:
        print(f"\n{model_name}")
        print("-" * 70)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        
        for device in devices:
            try:
                latency = profile_model(model, dummy_input, device)
                results[f"{model_name}_{device}"] = latency
                
                device_label = "GPU (CUDA)" if device == "cuda" else "CPU"
                print(f"\n  {device_label}:")
                print(f"    Mean:   {latency['mean_ms']:.4f} ms")
                print(f"    Median: {latency['median_ms']:.4f} ms")
                print(f"    P95:    {latency['p95_ms']:.4f} ms")
                print(f"    P99:    {latency['p99_ms']:.4f} ms")
                print(f"    Min:    {latency['min_ms']:.4f} ms")
                print(f"    Max:    {latency['max_ms']:.4f} ms")
                
            except Exception as e:
                print(f"  {device}: FAILED - {str(e)}")
    
    return results


def main():
    print("\n" + "=" * 70)
    print("JETSON EDGE DEVICE INFERENCE BENCHMARK")
    print("=" * 70)
    
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    results = benchmark_all()
    
    # Save results
    output_path = Path("/workspace/benchmarks/inference_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("DEPLOYMENT FEASIBILITY SUMMARY")
    print("=" * 70)
    
    # Jetson Nano typical specs: ~10-20ms per inference acceptable
    # Jetson Orin Nano: <5ms per inference achievable
    # Edge control loop: 0.1s (100ms) is typical
    
    for result_name, latency in results.items():
        mean_latency = latency["mean_ms"]
        model_type = "CNN-LSTM"
        device = "GPU" if "cuda" in result_name else "CPU"
        
        print(f"\n{model_type} on {device}:")
        print(f"  Avg latency: {mean_latency:.4f} ms")
        
        if mean_latency < 5:
            feasibility = "✅ Excellent - Real-time capable"
        elif mean_latency < 20:
            feasibility = "✅ Good - Suitable for 100ms control loop"
        elif mean_latency < 50:
            feasibility = "⚠️  Marginal - May cause control delays"
        else:
            feasibility = "❌ Poor - Not suitable for edge deployment"
        
        print(f"  Feasibility: {feasibility}")


if __name__ == "__main__":
    main()
