#!/usr/bin/env python3
"""Validate an existing processed dataset and generate quality reports.

Runs comprehensive quality checks on a dataset that has already been
processed by the pipeline, producing an HTML report with distribution
analysis, outlier detection, and completeness metrics.

Usage:
    python scripts/validate_dataset.py \\
        --dataset /data/ml_ready \\
        --format webdataset \\
        --report-output /reports/quality.html \\
        --max-samples 5000
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

from src.quality.data_quality import (
    DataQualityAnalyzer,
    DataQualityReport,
    check_patient_leakage,
)
from src.quality.report_generator import ReportGenerator
from src.dataloader.medical_dataloader import WebDatasetReader
from src.utils.logging_utils import setup_logging, ProgressTracker

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a processed medical imaging dataset",
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to processed dataset",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["webdataset", "hdf5", "numpy"],
        default="webdataset",
        help="Dataset format",
    )
    parser.add_argument(
        "--report-output", "-o",
        default="./reports/data_quality_report.html",
        help="Output path for HTML quality report",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to analyze (for quick validation)",
    )
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="Check for patient-level data leakage across splits",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Directory containing train/val/test split manifest files",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path for programmatic access",
    )
    return parser.parse_args()


def load_webdataset_samples(
    dataset_path: str,
    max_samples: Optional[int] = None,
) -> tuple:
    """Load samples from WebDataset shards for validation."""
    shard_paths = sorted(str(p) for p in Path(dataset_path).glob("*.tar"))
    if not shard_paths:
        logger.error("No tar shards found in %s", dataset_path)
        return [], [], [], []

    reader = WebDatasetReader(shard_paths, shuffle_shards=False)
    images, metadata_list, labels, keys = [], [], [], []

    tracker = ProgressTracker(
        total=max_samples or 999999,
        description="Loading samples",
    )

    for key, image, metadata in reader:
        images.append(image)
        metadata_list.append(metadata)
        keys.append(key)

        label = metadata.get("label")
        if label is not None:
            labels.append(label)

        tracker.update(1)

        if max_samples and len(images) >= max_samples:
            break

    tracker.close()
    return images, metadata_list, labels, keys


def load_hdf5_samples(
    dataset_path: str,
    max_samples: Optional[int] = None,
) -> tuple:
    """Load samples from HDF5 for validation."""
    from src.dataloader.medical_dataloader import HDF5Reader

    reader = HDF5Reader(dataset_path)
    reader.open()

    n = min(len(reader), max_samples or len(reader))
    images, metadata_list, labels, keys = [], [], [], []

    for i in range(n):
        key, image, metadata = reader[i]
        images.append(image)
        metadata_list.append(metadata)
        keys.append(key)
        label = metadata.get("label")
        if label is not None:
            labels.append(label)

    reader.close()
    return images, metadata_list, labels, keys


def check_splits_leakage(splits_dir: str) -> Dict[str, Any]:
    """Check for patient leakage across train/val/test splits."""
    def load_patient_ids(manifest_path: str) -> List[str]:
        with open(manifest_path) as f:
            manifest = json.load(f)
        return [
            entry.get("patient_id", "")
            for entry in manifest
            if entry.get("patient_id")
        ]

    splits_path = Path(splits_dir)
    train_ids = load_patient_ids(str(splits_path / "train_manifest.json"))
    val_ids = load_patient_ids(str(splits_path / "val_manifest.json"))
    test_path = splits_path / "test_manifest.json"
    test_ids = load_patient_ids(str(test_path)) if test_path.exists() else None

    return check_patient_leakage(train_ids, val_ids, test_ids)


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    logger.info("=" * 60)
    logger.info("Dataset Validation")
    logger.info("=" * 60)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Format: %s", args.format)

    t0 = time.monotonic()

    # Load samples
    logger.info("Loading samples...")
    if args.format == "webdataset":
        images, metadata_list, labels, keys = load_webdataset_samples(
            args.dataset, args.max_samples,
        )
    elif args.format == "hdf5":
        images, metadata_list, labels, keys = load_hdf5_samples(
            args.dataset, args.max_samples,
        )
    else:
        logger.error("Format %s not yet supported for validation", args.format)
        sys.exit(1)

    logger.info("Loaded %d samples in %.1fs", len(images), time.monotonic() - t0)

    if not images:
        logger.error("No samples loaded. Check dataset path and format.")
        sys.exit(1)

    # Run quality analysis
    logger.info("Running quality analysis...")
    analyzer = DataQualityAnalyzer(
        dataset_name=Path(args.dataset).name,
        completeness_threshold=0.95,
        outlier_method="iqr",
    )
    report = analyzer.analyze(
        images=images,
        metadata=metadata_list if metadata_list else None,
        labels=labels if labels else None,
        keys=keys,
    )

    # Patient leakage check
    if args.check_leakage and args.splits_dir:
        logger.info("Checking for patient-level data leakage...")
        leakage_result = check_splits_leakage(args.splits_dir)
        report.patient_leak_detected = leakage_result.get("has_leakage", False)
        if report.patient_leak_detected:
            report.warnings.append(
                f"Patient leakage detected! "
                f"Train-Val overlap: {leakage_result.get('train_val_overlap', 0)} patients"
            )

    # Generate HTML report
    logger.info("Generating HTML report...")
    generator = ReportGenerator(output_dir=str(Path(args.report_output).parent))
    report_path = generator.generate(
        report, filename=Path(args.report_output).name,
    )
    logger.info("HTML report saved to: %s", report_path)

    # Optional JSON export
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("JSON report saved to: %s", args.json_output)

    # Print summary
    total_time = time.monotonic() - t0
    logger.info("=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    logger.info("Total samples: %d", report.total_samples)
    if report.outlier_report:
        logger.info("Outliers: %d (%.1f%%)",
                     report.outlier_report.outlier_count,
                     report.outlier_report.outlier_fraction * 100)
    if report.completeness_report:
        n_missing = len(report.completeness_report.critical_missing)
        logger.info("Fields below completeness threshold: %d", n_missing)
    if report.class_balance:
        logger.info("Classes: %d, Imbalance ratio: %.1f:1",
                     report.class_balance.num_classes,
                     report.class_balance.imbalance_ratio)
    logger.info("Warnings: %d", len(report.warnings))
    for w in report.warnings:
        logger.warning("  - %s", w)
    logger.info("Total validation time: %.1fs", total_time)


if __name__ == "__main__":
    main()
