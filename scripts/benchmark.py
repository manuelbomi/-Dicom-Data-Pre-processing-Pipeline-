#!/usr/bin/env python3
"""Performance benchmarking script for the DICOM processing pipeline.

Measures throughput and scaling characteristics across different worker
counts and processing configurations.

Usage:
    python scripts/benchmark.py \\
        --input /data/raw_dicoms \\
        --max-workers 32 \\
        --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.dicom_scanner import DicomScanner
from src.ingestion.dicom_validator import DicomValidator
from src.preprocessing.pixel_processor import PixelProcessor, PixelProcessingConfig
from src.utils.logging_utils import setup_logging, MetricsCollector

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline performance benchmarks")
    parser.add_argument("--input", "-i", required=True, help="Input DICOM directory")
    parser.add_argument("--output", "-o", default="benchmark_results.json", help="Output JSON")
    parser.add_argument("--max-workers", type=int, default=16, help="Maximum worker count to test")
    parser.add_argument("--max-files", type=int, default=10000, help="Max files per benchmark")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup samples (excluded)")
    parser.add_argument("--repeat", type=int, default=3, help="Repetitions per benchmark")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def benchmark_scan(
    input_dir: str,
    worker_counts: List[int],
    max_files: int,
    repeat: int,
) -> List[Dict[str, Any]]:
    """Benchmark DICOM scanning throughput across worker counts."""
    results = []

    for n_workers in worker_counts:
        times = []
        files_found = 0

        for rep in range(repeat):
            scanner = DicomScanner(
                max_workers=n_workers,
                chunk_size=500,
                dedup_by_sop=False,
            )
            t0 = time.monotonic()
            result = scanner.scan(input_dir)
            duration = time.monotonic() - t0

            times.append(duration)
            files_found = result.dicom_files_found

            logger.info(
                "Scan benchmark: workers=%d, rep=%d, files=%d, time=%.2fs, throughput=%.0f files/s",
                n_workers, rep + 1, files_found, duration, result.throughput,
            )

        avg_time = np.mean(times)
        throughput = files_found / avg_time if avg_time > 0 else 0
        results.append({
            "stage": "scan",
            "workers": n_workers,
            "files": files_found,
            "avg_time_sec": round(float(avg_time), 3),
            "std_time_sec": round(float(np.std(times)), 3),
            "throughput_files_per_sec": round(throughput, 1),
            "times": [round(t, 3) for t in times],
        })

    return results


def benchmark_validation(
    input_dir: str,
    worker_counts: List[int],
    max_files: int,
    repeat: int,
) -> List[Dict[str, Any]]:
    """Benchmark DICOM validation throughput."""
    # First, get file list
    scanner = DicomScanner(max_workers=4, dedup_by_sop=False)
    scan_result = scanner.scan(input_dir)
    paths = [r.path for r in scan_result.records[:max_files]]

    if not paths:
        logger.warning("No files to validate")
        return []

    results = []
    for n_workers in worker_counts:
        times = []
        for rep in range(repeat):
            validator = DicomValidator(check_pixel_data=False)
            t0 = time.monotonic()
            validation_results = validator.validate_batch(paths)
            duration = time.monotonic() - t0
            times.append(duration)

            valid_count = sum(1 for r in validation_results if r.is_valid)
            logger.info(
                "Validation benchmark: workers=%d, rep=%d, files=%d, valid=%d, time=%.2fs",
                n_workers, rep + 1, len(paths), valid_count, duration,
            )

        avg_time = np.mean(times)
        throughput = len(paths) / avg_time if avg_time > 0 else 0
        results.append({
            "stage": "validation",
            "workers": n_workers,
            "files": len(paths),
            "avg_time_sec": round(float(avg_time), 3),
            "throughput_files_per_sec": round(throughput, 1),
            "times": [round(t, 3) for t in times],
        })

    return results


def compute_scaling_efficiency(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute scaling efficiency relative to single-worker baseline."""
    if not results:
        return results

    # Find single-worker baseline
    baseline = None
    for r in results:
        if r["workers"] == 1:
            baseline = r["throughput_files_per_sec"]
            break

    if baseline is None or baseline == 0:
        return results

    for r in results:
        ideal_throughput = baseline * r["workers"]
        actual_throughput = r["throughput_files_per_sec"]
        r["efficiency"] = round(actual_throughput / ideal_throughput * 100, 1) if ideal_throughput > 0 else 0
        r["speedup"] = round(actual_throughput / baseline, 2)

    return results


def generate_summary(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Generate benchmark summary with key metrics."""
    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
    }

    for stage_name, results in all_results.items():
        if not results:
            continue

        # Best throughput
        best = max(results, key=lambda r: r.get("throughput_files_per_sec", 0))
        single = next((r for r in results if r["workers"] == 1), results[0])

        summary["stages"][stage_name] = {
            "peak_throughput": best.get("throughput_files_per_sec", 0),
            "peak_workers": best.get("workers", 0),
            "single_worker_throughput": single.get("throughput_files_per_sec", 0),
            "max_speedup": best.get("speedup", 1.0),
            "max_efficiency": best.get("efficiency", 100.0),
        }

    return summary


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    logger.info("=" * 60)
    logger.info("DICOM Processing Pipeline - Performance Benchmark")
    logger.info("=" * 60)
    logger.info("Input: %s", args.input)
    logger.info("Max workers: %d", args.max_workers)
    logger.info("Max files: %d", args.max_files)
    logger.info("Repetitions: %d", args.repeat)

    # Worker counts to benchmark
    worker_counts = [1]
    w = 2
    while w <= args.max_workers:
        worker_counts.append(w)
        w *= 2
    if args.max_workers not in worker_counts:
        worker_counts.append(args.max_workers)

    logger.info("Worker counts to test: %s", worker_counts)

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    # Benchmark scan
    logger.info("-" * 40)
    logger.info("Benchmarking: DICOM Scanning")
    scan_results = benchmark_scan(args.input, worker_counts, args.max_files, args.repeat)
    scan_results = compute_scaling_efficiency(scan_results)
    all_results["scan"] = scan_results

    # Benchmark validation
    logger.info("-" * 40)
    logger.info("Benchmarking: DICOM Validation")
    val_results = benchmark_validation(args.input, [1], args.max_files, args.repeat)
    val_results = compute_scaling_efficiency(val_results)
    all_results["validation"] = val_results

    # Summary
    summary = generate_summary(all_results)

    # Write output
    output = {
        "summary": summary,
        "benchmarks": all_results,
        "config": {
            "input": args.input,
            "max_workers": args.max_workers,
            "max_files": args.max_files,
            "repeat": args.repeat,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to: %s", args.output)

    # Print summary
    logger.info("=" * 60)
    logger.info("Benchmark Summary")
    logger.info("=" * 60)
    for stage, metrics in summary.get("stages", {}).items():
        logger.info(
            "%s: peak=%.0f files/s @ %d workers (%.1fx speedup, %.1f%% efficiency)",
            stage,
            metrics["peak_throughput"],
            metrics["peak_workers"],
            metrics["max_speedup"],
            metrics["max_efficiency"],
        )


if __name__ == "__main__":
    main()
