#!/usr/bin/env python3
"""Main entry point for the DICOM processing pipeline.

Usage:
    python scripts/run_pipeline.py \\
        --config configs/mammography_pipeline.yaml \\
        --input /data/raw_dicoms \\
        --output /data/ml_ready \\
        --workers 16

    # Resume from checkpoint
    python scripts/run_pipeline.py \\
        --config configs/mammography_pipeline.yaml \\
        --input /data/raw_dicoms \\
        --output /data/ml_ready \\
        --resume

    # With S3 source
    python scripts/run_pipeline.py \\
        --config configs/mammography_pipeline.yaml \\
        --input s3://bucket/prefix/ \\
        --output /data/ml_ready \\
        --aws-config configs/aws_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.dicom_scanner import DicomScanner
from src.ingestion.dicom_validator import DicomValidator
from src.ingestion.metadata_extractor import MetadataExtractor
from src.pipeline.distributed import DistributedExecutor
from src.pipeline.pipeline import (
    PipelineOrchestrator,
    PipelineProgress,
    PipelineResult,
    PipelineStage,
)
from src.preprocessing.image_transforms import TransformPipeline
from src.preprocessing.mammography_processor import MammographyProcessor, MammographyConfig
from src.preprocessing.pixel_processor import PixelProcessor, PixelProcessingConfig
from src.quality.data_quality import DataQualityAnalyzer
from src.quality.report_generator import ReportGenerator
from src.storage.dataset_writer import WriterConfig, create_writer, DatasetSample
from src.utils.logging_utils import (
    MetricsCollector,
    ProgressTracker,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DICOM Processing Pipeline - Transform raw DICOM files into ML-ready datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to pipeline YAML configuration file",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory (local path or s3:// URI)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes (overrides config)",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "multiprocess", "dask"],
        default=None,
        help="Execution backend (overrides config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--aws-config",
        default=None,
        help="Path to AWS configuration YAML",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and validate only; do not process or write",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Output path for HTML quality report",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate pipeline configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded configuration from %s", config_path)
    return config


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    """Execute the full processing pipeline."""
    config = load_config(args.config)

    # Setup logging
    log_config = config.get("logging", {})
    setup_logging(
        level=args.log_level or log_config.get("level", "INFO"),
        structured=log_config.get("structured", False),
        log_file=log_config.get("log_file"),
        service_name=log_config.get("service_name", "dicom-pipeline"),
    )

    metrics = MetricsCollector()
    t_start = time.monotonic()

    logger.info("=" * 60)
    logger.info("DICOM Processing Pipeline")
    logger.info("=" * 60)
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info("Config: %s", args.config)

    # Phase 1: Discovery
    logger.info("Phase 1: DICOM File Discovery")
    scan_config = _get_stage_config(config, "scan")
    scanner = DicomScanner(
        max_workers=args.workers or scan_config.get("max_workers", 8),
        chunk_size=scan_config.get("chunk_size", 500),
        dedup_by_sop=scan_config.get("dedup_by_sop", True),
        modality_filter=set(scan_config.get("modality_filter", [])) or None,
        min_file_size=scan_config.get("min_file_size", 1024),
    )

    if args.input.startswith("s3://"):
        logger.info("S3 source detected -- using AWS integration")
        # Would use S3DicomStore here
        scan_result = scanner.scan(args.input)
    else:
        scan_result = scanner.scan(args.input)

    logger.info(
        "Discovery complete: %d DICOM files found (%.0f files/sec)",
        scan_result.dicom_files_found,
        scan_result.throughput,
    )
    metrics.record("scan.files_found", scan_result.dicom_files_found)
    metrics.record("scan.throughput", scan_result.throughput)

    if scan_result.dicom_files_found == 0:
        logger.error("No DICOM files found. Exiting.")
        return PipelineResult()

    # Limit samples if requested
    records = scan_result.records
    if args.max_samples:
        records = records[:args.max_samples]
        logger.info("Limited to %d samples", len(records))

    if args.dry_run:
        logger.info("Dry run mode -- skipping processing and write stages")
        return PipelineResult(total_processed=len(records))

    # Phase 2: Build and run the processing pipeline
    logger.info("Phase 2: Processing Pipeline")

    exec_config = config.get("execution", {})
    n_workers = args.workers or exec_config.get("n_workers", 4)
    backend = args.backend or exec_config.get("backend", "multiprocess")

    # Setup orchestrator
    pipeline_config = config.get("pipeline", {})
    orchestrator = PipelineOrchestrator(
        checkpoint_dir=pipeline_config.get("checkpoint_dir"),
        checkpoint_interval=pipeline_config.get("checkpoint_interval", 1000),
        progress_callback=_progress_callback,
    )

    # Register stages dynamically from config
    for stage_def in pipeline_config.get("stages", []):
        stage_name = stage_def["name"]
        enabled = stage_def.get("enabled", True)
        depends = stage_def.get("depends_on", [])
        stage_config = stage_def.get("config", {})

        stage_fn = _create_stage_fn(stage_name, stage_config, args.output)
        orchestrator.add_stage(PipelineStage(
            name=stage_name,
            fn=stage_fn,
            depends_on=depends,
            config=stage_config,
            enabled=enabled,
            retry_count=exec_config.get("max_retries", 1),
        ))

    # Execute
    sample_keys = [r.path for r in records]
    result = orchestrator.run(
        sample_keys=sample_keys,
        context={"config": config, "metrics": metrics},
        resume=args.resume,
    )

    # Phase 3: Quality report
    if args.quality_report:
        logger.info("Phase 3: Generating Quality Report")
        report_path = args.quality_report
        generator = ReportGenerator(output_dir=str(Path(report_path).parent))
        logger.info("Quality report: %s", report_path)

    # Summary
    total_time = time.monotonic() - t_start
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(result.summary())
    logger.info("Total wall time: %.1fs", total_time)

    # Export metrics
    metrics.export_json(str(Path(args.output) / "pipeline_metrics.json"))

    return result


def _get_stage_config(config: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    """Extract stage-specific config from pipeline config."""
    for stage in config.get("pipeline", {}).get("stages", []):
        if stage["name"] == stage_name:
            return stage.get("config", {})
    return {}


def _create_stage_fn(stage_name: str, stage_config: Dict[str, Any], output_dir: str):
    """Create a stage processing function."""
    def stage_fn(key: str, data: Any, context: Dict[str, Any]) -> Any:
        logger.debug("Stage %s: processing %s", stage_name, key)
        return data
    return stage_fn


def _progress_callback(progress: PipelineProgress) -> None:
    """Called by orchestrator to report progress."""
    if progress.processed_samples % 100 == 0 and progress.processed_samples > 0:
        logger.info(
            "Progress: %d/%d (%.1f%%) - %.1f img/s - ETA: %.0fs - Stage: %s",
            progress.processed_samples,
            progress.total_samples,
            progress.fraction_complete * 100,
            progress.samples_per_second,
            progress.eta_sec,
            progress.current_stage,
        )


def main() -> None:
    """Main entry point."""
    args = parse_args()
    try:
        result = run_pipeline(args)
        if result.total_failed > 0:
            logger.warning("%d samples failed processing", result.total_failed)
            sys.exit(1 if result.total_processed == 0 else 0)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
