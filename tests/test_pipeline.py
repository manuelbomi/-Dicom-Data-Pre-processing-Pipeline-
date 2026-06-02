"""Tests for the DICOM processing pipeline.

Covers core functionality with synthetic DICOM data to avoid
requiring actual medical images in the test environment.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root
import sys
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="dicom_pipeline_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Create a synthetic mammogram-like image."""
    np.random.seed(42)
    image = np.zeros((2048, 1024), dtype=np.float32)
    # Simulate breast region (left side)
    y, x = np.ogrid[:2048, :1024]
    breast_mask = ((x - 200) ** 2 / 300 ** 2 + (y - 1024) ** 2 / 900 ** 2) < 1
    image[breast_mask] = np.random.uniform(0.2, 0.8, size=breast_mask.sum()).astype(np.float32)
    # Add some noise
    image += np.random.normal(0, 0.02, image.shape).astype(np.float32)
    image = np.clip(image, 0, 1)
    return image


@pytest.fixture
def sample_metadata():
    """Create sample normalized metadata."""
    return {
        "sop_instance_uid": "1.2.3.4.5",
        "study_instance_uid": "1.2.3.4",
        "series_instance_uid": "1.2.3",
        "patient_id": "TEST_001",
        "modality": "MG",
        "manufacturer": "Hologic, Inc.",
        "manufacturer_normalized": "Hologic",
        "rows": 2048,
        "columns": 1024,
        "bits_allocated": 16,
        "bits_stored": 12,
        "image_laterality": "L",
        "view_position": "MLO",
        "pixel_spacing_mm": (0.07, 0.07),
    }


# ---------------------------------------------------------------------------
# Image Transform Tests
# ---------------------------------------------------------------------------

class TestImageTransforms:
    """Tests for image transform pipeline."""

    def test_resize(self, sample_image):
        from src.preprocessing.image_transforms import Resize
        transform = Resize(target_size=(512, 256))
        result = transform(sample_image)
        assert result.shape == (512, 256)

    def test_resize_preserve_aspect(self, sample_image):
        from src.preprocessing.image_transforms import Resize
        transform = Resize(target_size=(512, 512), preserve_aspect_ratio=True)
        result = transform(sample_image)
        # Aspect ratio preserved: 2048x1024 -> height limited
        assert result.shape[0] == 512 or result.shape[1] == 512

    def test_pad_to_size(self, sample_image):
        from src.preprocessing.image_transforms import PadToSize
        small = sample_image[:100, :100]
        transform = PadToSize(target_size=(200, 200), fill_value=0.0)
        result = transform(small)
        assert result.shape == (200, 200)
        # Check center is original
        assert result[50, 50] == small[0, 0]

    def test_center_crop(self, sample_image):
        from src.preprocessing.image_transforms import CenterCrop
        transform = CenterCrop(crop_size=(1024, 512))
        result = transform(sample_image)
        assert result.shape == (1024, 512)

    def test_resize_with_pad(self, sample_image):
        from src.preprocessing.image_transforms import ResizeWithPad
        transform = ResizeWithPad(target_size=(512, 512))
        result = transform(sample_image)
        assert result.shape == (512, 512)

    def test_intensity_normalize_zero_one(self, sample_image):
        from src.preprocessing.image_transforms import IntensityNormalize
        transform = IntensityNormalize(method="zero_one")
        result = transform(sample_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_intensity_normalize_percentile(self, sample_image):
        from src.preprocessing.image_transforms import IntensityNormalize
        transform = IntensityNormalize(method="percentile")
        result = transform(sample_image)
        assert result.min() >= -1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_intensity_normalize_zscore(self, sample_image):
        from src.preprocessing.image_transforms import IntensityNormalize
        transform = IntensityNormalize(method="z_score")
        result = transform(sample_image)
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.2

    def test_add_channel_dim(self, sample_image):
        from src.preprocessing.image_transforms import AddChannelDim
        transform = AddChannelDim()
        result = transform(sample_image)
        assert result.ndim == 3
        assert result.shape[0] == 1

    def test_transform_pipeline(self, sample_image):
        from src.preprocessing.image_transforms import (
            TransformPipeline, Resize, IntensityNormalize, AddChannelDim, ToTensor,
        )
        pipeline = TransformPipeline([
            Resize((256, 128)),
            IntensityNormalize(method="zero_one"),
            AddChannelDim(),
            ToTensor(),
        ])
        result = pipeline(sample_image)
        assert result.shape == (1, 256, 128)
        assert result.dtype == np.float32

    def test_transform_pipeline_from_config(self):
        from src.preprocessing.image_transforms import TransformPipeline
        config = {
            "transforms": [
                {"name": "Resize", "params": {"target_size": [256, 256]}},
                {"name": "IntensityNormalize", "params": {"method": "zero_one"}},
                {"name": "AddChannelDim", "params": {}},
            ]
        }
        pipeline = TransformPipeline.from_config(config)
        assert len(pipeline.transforms) == 3

    def test_mammography_defaults(self):
        from src.preprocessing.image_transforms import get_mammography_transforms
        pipeline = get_mammography_transforms(target_size=(512, 256))
        assert len(pipeline.transforms) == 4


# ---------------------------------------------------------------------------
# Mammography Processor Tests
# ---------------------------------------------------------------------------

class TestMammographyProcessor:
    """Tests for mammography-specific processing."""

    def test_detect_orientation_left(self):
        from src.preprocessing.mammography_processor import detect_orientation, Orientation
        image = np.zeros((100, 100), dtype=np.float32)
        image[:, :40] = 1.0  # Bright left side
        assert detect_orientation(image) == Orientation.LEFT_FACING

    def test_detect_orientation_right(self):
        from src.preprocessing.mammography_processor import detect_orientation, Orientation
        image = np.zeros((100, 100), dtype=np.float32)
        image[:, 60:] = 1.0  # Bright right side
        assert detect_orientation(image) == Orientation.RIGHT_FACING

    def test_normalize_laterality_flip(self):
        from src.preprocessing.mammography_processor import normalize_laterality, Orientation
        image = np.zeros((100, 100), dtype=np.float32)
        image[:, 60:] = 1.0  # Right-facing
        result, flipped = normalize_laterality(
            image, laterality="R", target=Orientation.LEFT_FACING,
        )
        assert flipped is True
        assert result[:, :40].mean() > result[:, 60:].mean()

    def test_normalize_laterality_no_flip(self):
        from src.preprocessing.mammography_processor import normalize_laterality, Orientation
        image = np.zeros((100, 100), dtype=np.float32)
        image[:, :40] = 1.0
        result, flipped = normalize_laterality(
            image, laterality="L", target=Orientation.LEFT_FACING,
        )
        assert flipped is False

    def test_segment_breast(self, sample_image):
        from src.preprocessing.mammography_processor import segment_breast
        mask = segment_breast(sample_image)
        assert mask.shape == sample_image.shape
        assert mask.dtype == np.uint8
        assert np.any(mask > 0)

    def test_apply_clahe(self, sample_image):
        from src.preprocessing.mammography_processor import apply_clahe
        result = apply_clahe(sample_image, clip_limit=2.0)
        assert result.shape == sample_image.shape
        assert result.dtype == np.float32
        assert result.max() <= 1.0 + 1e-6

    def test_crop_to_breast(self, sample_image):
        from src.preprocessing.mammography_processor import crop_to_breast_region
        mask = np.zeros_like(sample_image, dtype=np.uint8)
        mask[100:500, 50:200] = 255
        cropped, bbox = crop_to_breast_region(sample_image, mask, padding=5)
        assert cropped.shape[0] < sample_image.shape[0]
        assert len(bbox) == 4

    def test_full_processor(self, sample_image):
        from src.preprocessing.mammography_processor import (
            MammographyProcessor, MammographyConfig,
        )
        config = MammographyConfig(
            clahe_enabled=True,
            pectoral_removal_enabled=False,
            crop_to_breast=True,
        )
        processor = MammographyProcessor(config)
        result = processor.process(
            image=sample_image,
            laterality="L",
            view_position="CC",
        )
        assert "image" in result
        assert "analysis" in result
        assert result["image"].ndim == 2


# ---------------------------------------------------------------------------
# Pixel Processor Tests
# ---------------------------------------------------------------------------

class TestPixelProcessor:
    """Tests for pixel data processing."""

    def test_normalize_bit_depth_min_max(self):
        from src.preprocessing.pixel_processor import normalize_bit_depth, PixelProcessingConfig, NormalizationMethod
        arr = np.array([0, 1023, 2047, 4095], dtype=np.uint16)
        config = PixelProcessingConfig(normalization=NormalizationMethod.MIN_MAX)
        result = normalize_bit_depth(arr, bits_stored=12, config=config)
        assert result.min() >= 0.0
        assert abs(result.max() - 1.0) < 0.01

    def test_normalize_bit_depth_fixed_range(self):
        from src.preprocessing.pixel_processor import normalize_bit_depth, PixelProcessingConfig, NormalizationMethod
        arr = np.array([0, 2048, 4095], dtype=np.uint16)
        config = PixelProcessingConfig(normalization=NormalizationMethod.FIXED_RANGE)
        result = normalize_bit_depth(arr, bits_stored=12, config=config)
        assert abs(result.max() - 1.0) < 0.01

    def test_handle_photometric_interpretation(self):
        from src.preprocessing.pixel_processor import handle_photometric_interpretation, PhotometricInterpretation
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = handle_photometric_interpretation(
            arr, "MONOCHROME1", PhotometricInterpretation.MONOCHROME2,
        )
        assert result[0] == 1.0  # Was 0 (white), now inverted
        assert result[2] == 0.0


# ---------------------------------------------------------------------------
# Data Quality Tests
# ---------------------------------------------------------------------------

class TestDataQuality:
    """Tests for data quality analysis."""

    def test_distribution_stats(self):
        from src.quality.data_quality import compute_distribution_stats
        values = np.random.normal(0.5, 0.1, 1000)
        stats = compute_distribution_stats(values)
        assert abs(stats.mean - 0.5) < 0.05
        assert stats.count == 1000
        assert stats.histogram_bins is not None

    def test_outlier_detection_iqr(self):
        from src.quality.data_quality import detect_outliers_iqr
        values = np.concatenate([
            np.random.normal(0.5, 0.1, 990),
            np.array([5.0] * 10),  # Outliers
        ])
        report = detect_outliers_iqr(values)
        assert report.outlier_count > 0
        assert report.outlier_fraction > 0

    def test_outlier_detection_zscore(self):
        from src.quality.data_quality import detect_outliers_zscore
        values = np.concatenate([
            np.random.normal(0, 1, 990),
            np.array([10.0] * 10),
        ])
        report = detect_outliers_zscore(values, threshold=3.0)
        assert report.outlier_count >= 10

    def test_class_balance(self):
        from src.quality.data_quality import analyze_class_balance
        labels = [0] * 900 + [1] * 100  # 9:1 imbalance
        report = analyze_class_balance(labels)
        assert report.num_classes == 2
        assert report.imbalance_ratio == 9.0
        assert not report.is_severely_imbalanced  # 9 < 10

    def test_class_balance_severe(self):
        from src.quality.data_quality import analyze_class_balance
        labels = [0] * 950 + [1] * 50  # 19:1 imbalance
        report = analyze_class_balance(labels)
        assert report.is_severely_imbalanced

    def test_metadata_completeness(self):
        from src.quality.data_quality import check_metadata_completeness
        records = [
            {"patient_id": "P1", "modality": "MG", "manufacturer": "Hologic"},
            {"patient_id": "P2", "modality": "MG", "manufacturer": None},
            {"patient_id": "P3", "modality": "MG"},
        ]
        report = check_metadata_completeness(
            records,
            required_fields=["patient_id", "modality", "manufacturer"],
            threshold=0.9,
        )
        assert report.field_completeness["patient_id"] == 1.0
        assert report.field_completeness["manufacturer"] < 1.0

    def test_patient_leakage_detection(self):
        from src.quality.data_quality import check_patient_leakage
        train = ["P1", "P2", "P3", "P4"]
        val = ["P3", "P5"]  # P3 is in both!
        result = check_patient_leakage(train, val)
        assert result["has_leakage"] is True
        assert result["train_val_overlap"] == 1


# ---------------------------------------------------------------------------
# Dataset Writer Tests
# ---------------------------------------------------------------------------

class TestDatasetWriter:
    """Tests for dataset writing."""

    def test_webdataset_writer(self, temp_dir):
        from src.storage.dataset_writer import WebDatasetWriter, WriterConfig, DatasetSample

        config = WriterConfig(
            output_dir=temp_dir,
            format="webdataset",
            shard_max_samples=5,
            shard_prefix="test",
        )
        writer = WebDatasetWriter(config)
        writer.open()

        for i in range(10):
            sample = DatasetSample(
                key=f"sample_{i:04d}",
                image=np.random.rand(1, 64, 64).astype(np.float32),
                label=i % 2,
                metadata={"idx": i},
            )
            writer.write(sample)

        stats = writer.close()
        assert stats.total_samples == 10
        assert stats.total_shards >= 2

        # Check tar files exist
        tar_files = list(Path(temp_dir).glob("*.tar"))
        assert len(tar_files) >= 2

        # Check manifest
        manifest_path = Path(temp_dir) / "manifest.json"
        assert manifest_path.exists()

    def test_create_writer_factory(self, temp_dir):
        from src.storage.dataset_writer import create_writer, WriterConfig
        config = WriterConfig(output_dir=temp_dir, format="webdataset")
        writer = create_writer(config)
        assert writer is not None


# ---------------------------------------------------------------------------
# Pipeline Orchestrator Tests
# ---------------------------------------------------------------------------

class TestPipelineOrchestrator:
    """Tests for pipeline orchestration."""

    def test_simple_pipeline(self):
        from src.pipeline.pipeline import PipelineOrchestrator, PipelineStage

        def stage_a(key, data, ctx):
            return (data or 0) + 1

        def stage_b(key, data, ctx):
            return data * 2

        orchestrator = PipelineOrchestrator(stages=[
            PipelineStage("a", stage_a),
            PipelineStage("b", stage_b, depends_on=["a"]),
        ])

        result = orchestrator.run(
            sample_keys=["s1", "s2", "s3"],
            input_data={"s1": 10, "s2": 20, "s3": 30},
        )

        assert result.total_processed == 3
        assert result.total_failed == 0

    def test_pipeline_with_failure(self):
        from src.pipeline.pipeline import PipelineOrchestrator, PipelineStage

        def failing_stage(key, data, ctx):
            if key == "bad":
                raise ValueError("Test error")
            return data

        orchestrator = PipelineOrchestrator(stages=[
            PipelineStage("process", failing_stage),
        ])

        result = orchestrator.run(sample_keys=["good", "bad", "good2"])
        assert result.total_processed == 2
        assert result.total_failed == 1
        assert len(result.errors) == 1

    def test_pipeline_checkpoint(self, temp_dir):
        from src.pipeline.pipeline import PipelineOrchestrator, PipelineStage, Checkpoint

        checkpoint = Checkpoint(temp_dir, interval_samples=2)

        def noop(key, data, ctx):
            return data

        orchestrator = PipelineOrchestrator(
            stages=[PipelineStage("noop", noop)],
            checkpoint_dir=temp_dir,
            checkpoint_interval=2,
        )

        result = orchestrator.run(sample_keys=["a", "b", "c", "d", "e"])
        assert result.total_processed == 5

        # Checkpoint should exist
        assert Path(temp_dir, "latest.pkl").exists()

    def test_stage_ordering(self):
        from src.pipeline.pipeline import PipelineOrchestrator, PipelineStage
        order = []

        def make_fn(name):
            def fn(key, data, ctx):
                order.append(name)
                return data
            return fn

        orchestrator = PipelineOrchestrator(stages=[
            PipelineStage("c", make_fn("c"), depends_on=["b"]),
            PipelineStage("a", make_fn("a")),
            PipelineStage("b", make_fn("b"), depends_on=["a"]),
        ])

        result = orchestrator.run(sample_keys=["x"])
        assert order == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Distributed Executor Tests
# ---------------------------------------------------------------------------

class TestDistributedExecutor:
    """Tests for distributed execution backends."""

    def test_local_executor(self):
        from src.pipeline.distributed import LocalExecutor

        executor = LocalExecutor()
        results = executor.map(
            lambda x: x ** 2,
            [1, 2, 3, 4, 5],
            keys=["a", "b", "c", "d", "e"],
        )
        assert len(results) == 5
        assert all(r.success for r in results)
        assert results[0].output == 1
        assert results[4].output == 25

    def test_local_executor_with_errors(self):
        from src.pipeline.distributed import LocalExecutor

        def sometimes_fail(x):
            if x == 3:
                raise ValueError("bad")
            return x

        executor = LocalExecutor()
        results = executor.map(sometimes_fail, [1, 2, 3, 4])
        assert results[2].success is False
        assert "bad" in results[2].error


# ---------------------------------------------------------------------------
# Logging / Metrics Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for metrics collection."""

    def test_metrics_collector(self):
        from src.utils.logging_utils import MetricsCollector
        m = MetricsCollector()
        m.record("latency_ms", 10.0)
        m.record("latency_ms", 20.0)
        m.record("latency_ms", 30.0)

        s = m.get_summary("latency_ms")
        assert s is not None
        assert s.count == 3
        assert abs(s.mean - 20.0) < 0.01
        assert s.min == 10.0
        assert s.max == 30.0

    def test_metrics_counter(self):
        from src.utils.logging_utils import MetricsCollector
        m = MetricsCollector()
        m.increment("images_processed", 5)
        m.increment("images_processed", 3)
        assert m.get_counter("images_processed") == 8

    def test_metrics_timer(self):
        from src.utils.logging_utils import MetricsCollector
        import time
        m = MetricsCollector()
        with m.timer("test_op"):
            time.sleep(0.01)
        s = m.get_summary("test_op")
        assert s is not None
        assert s.last > 0

    def test_metrics_export(self, temp_dir):
        from src.utils.logging_utils import MetricsCollector
        m = MetricsCollector()
        m.record("throughput", 100.0)
        path = os.path.join(temp_dir, "metrics.json")
        m.export_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "summaries" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
