"""Data quality analysis for medical imaging datasets.

Provides comprehensive quality checks on processed datasets to catch
issues before expensive ML training runs. Includes:

  - **Distribution analysis**: Pixel intensity distributions, image size
    distributions, metadata value distributions.
  - **Outlier detection**: Identify anomalous images via statistical
    methods (IQR, z-score, isolation forest).
  - **Missing data reports**: Completeness analysis for metadata fields.
  - **Class balance analysis**: Label distribution and potential bias detection.
  - **Cross-validation checks**: Verify patient-level splits don't leak.

These checks are essential for medical imaging where data quality directly
impacts model safety and regulatory compliance.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DistributionStats:
    """Statistical summary of a distribution."""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    histogram_bins: Optional[List[float]] = None
    histogram_counts: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class OutlierReport:
    """Report on detected outliers."""

    method: str  # "iqr", "zscore", "isolation_forest"
    total_samples: int = 0
    outlier_count: int = 0
    outlier_fraction: float = 0.0
    outlier_indices: List[int] = field(default_factory=list)
    outlier_keys: List[str] = field(default_factory=list)
    outlier_values: List[float] = field(default_factory=list)
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None


@dataclass
class CompletenessReport:
    """Report on metadata field completeness."""

    total_samples: int = 0
    field_completeness: Dict[str, float] = field(default_factory=dict)  # field -> fraction present
    field_missing_count: Dict[str, int] = field(default_factory=dict)
    critical_missing: List[str] = field(default_factory=list)  # Fields below threshold
    completeness_threshold: float = 0.95


@dataclass
class ClassBalanceReport:
    """Report on label/class distribution."""

    total_samples: int = 0
    num_classes: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_fractions: Dict[str, float] = field(default_factory=dict)
    majority_class: str = ""
    minority_class: str = ""
    imbalance_ratio: float = 0.0  # max_count / min_count
    effective_num_samples: float = 0.0
    entropy: float = 0.0
    is_severely_imbalanced: bool = False  # ratio > 10:1


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""

    dataset_name: str = ""
    total_samples: int = 0
    timestamp: str = ""

    # Per-feature distributions
    intensity_stats: Optional[DistributionStats] = None
    image_height_stats: Optional[DistributionStats] = None
    image_width_stats: Optional[DistributionStats] = None
    file_size_stats: Optional[DistributionStats] = None

    # Metadata distributions
    modality_distribution: Dict[str, int] = field(default_factory=dict)
    manufacturer_distribution: Dict[str, int] = field(default_factory=dict)
    laterality_distribution: Dict[str, int] = field(default_factory=dict)
    view_distribution: Dict[str, int] = field(default_factory=dict)

    # Quality checks
    outlier_report: Optional[OutlierReport] = None
    completeness_report: Optional[CompletenessReport] = None
    class_balance: Optional[ClassBalanceReport] = None

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Data splits (if applicable)
    patient_leak_detected: bool = False
    duplicate_images_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to nested dictionary for serialization."""
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                result[key] = value
            elif isinstance(value, list):
                result[key] = value
            else:
                result[key] = value
        return result


def compute_distribution_stats(
    values: np.ndarray,
    n_bins: int = 50,
) -> DistributionStats:
    """Compute comprehensive distribution statistics.

    Args:
        values: 1D array of values.
        n_bins: Number of histogram bins.

    Returns:
        DistributionStats with all computed metrics.
    """
    if len(values) == 0:
        return DistributionStats()

    values = values.astype(np.float64)
    stats = DistributionStats(
        count=len(values),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        p5=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
    )

    # Skewness and kurtosis
    if stats.std > 1e-10:
        centered = values - stats.mean
        stats.skewness = float(np.mean(centered ** 3) / (stats.std ** 3))
        stats.kurtosis = float(np.mean(centered ** 4) / (stats.std ** 4) - 3.0)

    # Histogram
    counts, bin_edges = np.histogram(values, bins=n_bins)
    stats.histogram_bins = [float(b) for b in bin_edges]
    stats.histogram_counts = [int(c) for c in counts]

    return stats


def detect_outliers_iqr(
    values: np.ndarray,
    keys: Optional[List[str]] = None,
    multiplier: float = 1.5,
) -> OutlierReport:
    """Detect outliers using the Interquartile Range (IQR) method.

    Points outside [Q1 - k*IQR, Q3 + k*IQR] are flagged as outliers.
    IQR is robust to extreme values, making it suitable for medical
    imaging data which often has heavy-tailed distributions.

    Args:
        values: 1D array of values to check.
        keys: Optional sample identifiers.
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme only).

    Returns:
        OutlierReport with detected outliers.
    """
    report = OutlierReport(method="iqr", total_samples=len(values))

    if len(values) < 4:
        return report

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1

    report.threshold_low = q1 - multiplier * iqr
    report.threshold_high = q3 + multiplier * iqr

    outlier_mask = (values < report.threshold_low) | (values > report.threshold_high)
    outlier_indices = np.where(outlier_mask)[0]

    report.outlier_count = int(len(outlier_indices))
    report.outlier_fraction = report.outlier_count / max(report.total_samples, 1)
    report.outlier_indices = outlier_indices.tolist()
    report.outlier_values = [float(values[i]) for i in outlier_indices[:100]]

    if keys:
        report.outlier_keys = [keys[i] for i in outlier_indices[:100]]

    return report


def detect_outliers_zscore(
    values: np.ndarray,
    keys: Optional[List[str]] = None,
    threshold: float = 3.0,
) -> OutlierReport:
    """Detect outliers using z-score method.

    Args:
        values: 1D array of values.
        keys: Optional sample identifiers.
        threshold: Z-score threshold for outlier detection.

    Returns:
        OutlierReport.
    """
    report = OutlierReport(method="zscore", total_samples=len(values))

    if len(values) < 3:
        return report

    mean = float(np.mean(values))
    std = float(np.std(values))

    if std < 1e-10:
        return report

    z_scores = np.abs((values - mean) / std)
    outlier_mask = z_scores > threshold
    outlier_indices = np.where(outlier_mask)[0]

    report.outlier_count = int(len(outlier_indices))
    report.outlier_fraction = report.outlier_count / max(report.total_samples, 1)
    report.outlier_indices = outlier_indices.tolist()
    report.outlier_values = [float(values[i]) for i in outlier_indices[:100]]

    if keys:
        report.outlier_keys = [keys[i] for i in outlier_indices[:100]]

    return report


def check_metadata_completeness(
    metadata_records: List[Dict[str, Any]],
    required_fields: Optional[List[str]] = None,
    threshold: float = 0.95,
) -> CompletenessReport:
    """Analyze metadata field completeness across the dataset.

    Args:
        metadata_records: List of metadata dictionaries.
        required_fields: Fields that should be present in all records.
        threshold: Completeness threshold below which a field is flagged.

    Returns:
        CompletenessReport.
    """
    report = CompletenessReport(
        total_samples=len(metadata_records),
        completeness_threshold=threshold,
    )

    if not metadata_records:
        return report

    # Collect all fields
    all_fields: Set[str] = set()
    for record in metadata_records:
        all_fields.update(record.keys())

    # If required fields not specified, check all
    fields_to_check = required_fields or sorted(all_fields)

    for field_name in fields_to_check:
        present_count = sum(
            1 for r in metadata_records
            if field_name in r
            and r[field_name] is not None
            and str(r[field_name]).strip() != ""
        )
        missing = report.total_samples - present_count
        completeness = present_count / max(report.total_samples, 1)

        report.field_completeness[field_name] = round(completeness, 4)
        report.field_missing_count[field_name] = missing

        if completeness < threshold:
            report.critical_missing.append(field_name)

    return report


def analyze_class_balance(
    labels: List[Any],
    class_names: Optional[Dict[Any, str]] = None,
    severe_imbalance_ratio: float = 10.0,
) -> ClassBalanceReport:
    """Analyze class distribution for label imbalance.

    Label imbalance is a critical concern in medical imaging where
    positive findings (e.g., malignant lesions) are rare compared
    to normal cases.

    Args:
        labels: List of class labels.
        class_names: Optional mapping from label to display name.
        severe_imbalance_ratio: Threshold for flagging severe imbalance.

    Returns:
        ClassBalanceReport.
    """
    report = ClassBalanceReport(total_samples=len(labels))

    if not labels:
        return report

    counter = Counter(labels)
    report.num_classes = len(counter)

    for label, count in counter.most_common():
        name = class_names.get(label, str(label)) if class_names else str(label)
        report.class_counts[name] = count
        report.class_fractions[name] = round(count / report.total_samples, 4)

    # Majority and minority
    sorted_classes = counter.most_common()
    report.majority_class = str(sorted_classes[0][0])
    report.minority_class = str(sorted_classes[-1][0])

    max_count = sorted_classes[0][1]
    min_count = sorted_classes[-1][1]
    report.imbalance_ratio = max_count / max(min_count, 1)
    report.is_severely_imbalanced = report.imbalance_ratio > severe_imbalance_ratio

    # Effective number of samples (Cui et al. 2019)
    beta = 0.9999
    report.effective_num_samples = sum(
        (1.0 - beta ** count) / (1.0 - beta) for count in counter.values()
    )

    # Shannon entropy
    probs = np.array([c / report.total_samples for c in counter.values()])
    report.entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

    return report


def check_patient_leakage(
    train_patient_ids: List[str],
    val_patient_ids: List[str],
    test_patient_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check for patient-level data leakage between splits.

    In medical imaging, the same patient may have multiple studies/images.
    Training and validation/test sets must be split at the patient level
    to avoid information leakage.

    Args:
        train_patient_ids: Patient IDs in training set.
        val_patient_ids: Patient IDs in validation set.
        test_patient_ids: Patient IDs in test set (optional).

    Returns:
        Dictionary with leakage findings.
    """
    train_set = set(train_patient_ids)
    val_set = set(val_patient_ids)

    train_val_leak = train_set & val_set
    result = {
        "train_patients": len(train_set),
        "val_patients": len(val_set),
        "train_val_overlap": len(train_val_leak),
        "train_val_leaked_ids": sorted(list(train_val_leak))[:20],
        "has_leakage": len(train_val_leak) > 0,
    }

    if test_patient_ids is not None:
        test_set = set(test_patient_ids)
        train_test_leak = train_set & test_set
        val_test_leak = val_set & test_set
        result.update({
            "test_patients": len(test_set),
            "train_test_overlap": len(train_test_leak),
            "val_test_overlap": len(val_test_leak),
            "train_test_leaked_ids": sorted(list(train_test_leak))[:20],
            "has_leakage": result["has_leakage"] or len(train_test_leak) > 0 or len(val_test_leak) > 0,
        })

    if result["has_leakage"]:
        logger.warning(
            "Patient leakage detected! Train-Val: %d, Train-Test: %s",
            len(train_val_leak),
            result.get("train_test_overlap", "N/A"),
        )

    return result


class DataQualityAnalyzer:
    """Comprehensive data quality analyzer for medical imaging datasets.

    Runs all quality checks and produces a unified DataQualityReport.

    Args:
        dataset_name: Name for the report.
        required_metadata_fields: Fields that must be present.
        completeness_threshold: Threshold for metadata completeness.
        outlier_method: Method for outlier detection ("iqr" or "zscore").

    Example::

        analyzer = DataQualityAnalyzer(dataset_name="mammography_v2")
        report = analyzer.analyze(
            images=images,           # List of numpy arrays
            metadata=metadata_list,  # List of dicts
            labels=labels,           # List of labels
            keys=sop_uids,           # List of identifiers
        )
        print(f"Outliers: {report.outlier_report.outlier_count}")
        print(f"Missing critical fields: {report.completeness_report.critical_missing}")
    """

    def __init__(
        self,
        dataset_name: str = "",
        required_metadata_fields: Optional[List[str]] = None,
        completeness_threshold: float = 0.95,
        outlier_method: str = "iqr",
    ) -> None:
        self.dataset_name = dataset_name
        self.required_metadata_fields = required_metadata_fields or [
            "patient_id", "study_instance_uid", "modality",
            "manufacturer", "rows", "columns", "bits_stored",
            "image_laterality", "view_position",
        ]
        self.completeness_threshold = completeness_threshold
        self.outlier_method = outlier_method

    def analyze(
        self,
        images: Optional[List[np.ndarray]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        labels: Optional[List[Any]] = None,
        keys: Optional[List[str]] = None,
    ) -> DataQualityReport:
        """Run all quality analyses.

        Args:
            images: List of image arrays (can be None if only checking metadata).
            metadata: List of metadata dictionaries.
            labels: List of labels.
            keys: List of sample identifiers.

        Returns:
            Comprehensive DataQualityReport.
        """
        from datetime import datetime

        report = DataQualityReport(
            dataset_name=self.dataset_name,
            total_samples=len(images) if images else len(metadata or []),
            timestamp=datetime.now().isoformat(),
        )

        # Image statistics
        if images:
            self._analyze_images(images, keys, report)

        # Metadata completeness
        if metadata:
            self._analyze_metadata(metadata, report)

        # Class balance
        if labels:
            report.class_balance = analyze_class_balance(labels)
            if report.class_balance.is_severely_imbalanced:
                report.warnings.append(
                    f"Severe class imbalance detected: "
                    f"{report.class_balance.imbalance_ratio:.1f}:1 ratio"
                )
                report.recommendations.append(
                    "Consider oversampling minority class or using weighted loss"
                )

        # Generate recommendations
        self._generate_recommendations(report)

        return report

    def _analyze_images(
        self,
        images: List[np.ndarray],
        keys: Optional[List[str]],
        report: DataQualityReport,
    ) -> None:
        """Analyze image pixel distributions and dimensions."""
        mean_intensities = np.array([img.mean() for img in images])
        report.intensity_stats = compute_distribution_stats(mean_intensities)

        heights = np.array([img.shape[0] for img in images])
        widths = np.array([img.shape[-1] for img in images])
        report.image_height_stats = compute_distribution_stats(heights)
        report.image_width_stats = compute_distribution_stats(widths)

        # Outlier detection on mean intensity
        if self.outlier_method == "iqr":
            report.outlier_report = detect_outliers_iqr(mean_intensities, keys)
        else:
            report.outlier_report = detect_outliers_zscore(mean_intensities, keys)

        if report.outlier_report.outlier_fraction > 0.05:
            report.warnings.append(
                f"High outlier fraction: {report.outlier_report.outlier_fraction:.1%} "
                f"of images flagged as intensity outliers"
            )

    def _analyze_metadata(
        self,
        metadata: List[Dict[str, Any]],
        report: DataQualityReport,
    ) -> None:
        """Analyze metadata completeness and distributions."""
        report.completeness_report = check_metadata_completeness(
            metadata,
            required_fields=self.required_metadata_fields,
            threshold=self.completeness_threshold,
        )

        if report.completeness_report.critical_missing:
            report.warnings.append(
                f"Low completeness for fields: "
                f"{', '.join(report.completeness_report.critical_missing)}"
            )

        # Categorical distributions
        for field_name, report_field in [
            ("modality", "modality_distribution"),
            ("manufacturer_normalized", "manufacturer_distribution"),
            ("image_laterality", "laterality_distribution"),
            ("view_position", "view_distribution"),
        ]:
            counter = Counter(
                str(r.get(field_name, "Unknown"))
                for r in metadata
                if r.get(field_name)
            )
            setattr(report, report_field, dict(counter.most_common()))

    def _generate_recommendations(self, report: DataQualityReport) -> None:
        """Generate actionable recommendations based on findings."""
        if report.total_samples < 1000:
            report.recommendations.append(
                "Dataset is small (<1000 samples). Consider data augmentation "
                "and transfer learning from larger datasets."
            )

        if report.intensity_stats and report.intensity_stats.std < 0.01:
            report.recommendations.append(
                "Very low intensity variance. Check that normalization "
                "is applied correctly."
            )
