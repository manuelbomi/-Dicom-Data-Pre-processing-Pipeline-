"""Write processed medical images to efficient ML-ready formats.

Supports three output formats optimized for different use cases:

  - **HDF5**: Best for random access during experimentation. Single-file
    storage with compression. Suitable for datasets that fit on one machine.
  - **WebDataset (tar shards)**: Best for large-scale distributed training.
    Sequential-access tar archives that work with streaming DataLoaders.
    Each shard is a self-contained tar file with images and metadata.
  - **LMDB**: Best for fast random access on SSD. Memory-mapped key-value
    store with zero-copy reads.

All formats store both the processed image arrays and associated metadata
(labels, clinical features, processing provenance).
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import tarfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WriterConfig:
    """Configuration for dataset writers."""

    output_dir: str = "./output"
    format: str = "webdataset"  # "hdf5", "webdataset", "lmdb"

    # Sharding
    shard_size_mb: int = 256
    shard_max_samples: int = 1000
    shard_prefix: str = "shard"

    # Compression
    compression: str = "gzip"  # "gzip", "lz4", "none"
    compression_level: int = 4

    # HDF5-specific
    hdf5_chunk_shape: Optional[Tuple[int, ...]] = None
    hdf5_single_file: bool = True

    # LMDB-specific
    lmdb_map_size_gb: int = 100

    # Image encoding for WebDataset
    image_format: str = "npy"  # "npy", "png", "npz"

    # Metadata
    write_manifest: bool = True
    write_stats: bool = True


@dataclass
class DatasetSample:
    """A single sample to write to the dataset."""

    key: str  # Unique identifier (e.g., SOPInstanceUID)
    image: np.ndarray  # Processed image array
    label: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def size_bytes(self) -> int:
        return self.image.nbytes


@dataclass
class WriteStats:
    """Statistics from a write operation."""

    total_samples: int = 0
    total_bytes: int = 0
    total_shards: int = 0
    write_duration_sec: float = 0.0
    samples_per_second: float = 0.0
    bytes_per_second: float = 0.0
    output_paths: List[str] = field(default_factory=list)


class BaseWriter(ABC):
    """Abstract base class for dataset writers."""

    def __init__(self, config: WriterConfig) -> None:
        self.config = config
        self._stats = WriteStats()
        self._start_time: float = 0.0

    @abstractmethod
    def open(self) -> None:
        """Initialize the writer and create output directories."""

    @abstractmethod
    def write(self, sample: DatasetSample) -> None:
        """Write a single sample."""

    @abstractmethod
    def close(self) -> WriteStats:
        """Finalize and close the writer. Returns stats."""

    def __enter__(self) -> "BaseWriter":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class HDF5Writer(BaseWriter):
    """Write processed datasets to HDF5 format.

    Creates an HDF5 file with datasets:
      - "images": (N, C, H, W) float32 array
      - "labels": (N,) array
      - "metadata": JSON-encoded string array
      - "keys": string array of sample identifiers

    Supports chunked storage with compression for efficient I/O.
    """

    def __init__(self, config: WriterConfig) -> None:
        super().__init__(config)
        self._file = None
        self._images: List[np.ndarray] = []
        self._labels: List[Any] = []
        self._metadata: List[str] = []
        self._keys: List[str] = []
        self._shard_idx = 0

    def open(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._start_time = time.monotonic()
        logger.info("HDF5Writer opened: %s", self.config.output_dir)

    def write(self, sample: DatasetSample) -> None:
        self._images.append(sample.image)
        self._labels.append(sample.label if sample.label is not None else -1)
        self._metadata.append(json.dumps(sample.metadata or {}))
        self._keys.append(sample.key)
        self._stats.total_samples += 1
        self._stats.total_bytes += sample.size_bytes

        # Check if shard is full
        if len(self._images) >= self.config.shard_max_samples:
            self._flush_shard()

    def _flush_shard(self) -> None:
        """Write accumulated samples to an HDF5 file."""
        if not self._images:
            return

        import h5py

        if self.config.hdf5_single_file:
            path = os.path.join(self.config.output_dir, "dataset.h5")
            mode = "a" if os.path.exists(path) else "w"
        else:
            path = os.path.join(
                self.config.output_dir,
                f"{self.config.shard_prefix}_{self._shard_idx:06d}.h5",
            )
            mode = "w"

        compression = self.config.compression if self.config.compression != "none" else None

        with h5py.File(path, mode) as f:
            images = np.stack(self._images, axis=0)
            group_name = f"shard_{self._shard_idx:06d}"
            grp = f.create_group(group_name)

            grp.create_dataset(
                "images",
                data=images,
                chunks=self.config.hdf5_chunk_shape or True,
                compression=compression,
                compression_opts=self.config.compression_level if compression == "gzip" else None,
            )
            grp.create_dataset("labels", data=np.array(self._labels))

            dt = h5py.string_dtype()
            grp.create_dataset("metadata", data=self._metadata, dtype=dt)
            grp.create_dataset("keys", data=self._keys, dtype=dt)

        self._stats.output_paths.append(path)
        self._stats.total_shards += 1
        logger.info(
            "HDF5 shard %d written: %d samples -> %s",
            self._shard_idx,
            len(self._images),
            path,
        )

        self._images.clear()
        self._labels.clear()
        self._metadata.clear()
        self._keys.clear()
        self._shard_idx += 1

    def close(self) -> WriteStats:
        self._flush_shard()
        duration = time.monotonic() - self._start_time
        self._stats.write_duration_sec = duration
        if duration > 0:
            self._stats.samples_per_second = self._stats.total_samples / duration
            self._stats.bytes_per_second = self._stats.total_bytes / duration

        if self.config.write_manifest:
            self._write_manifest()
        return self._stats

    def _write_manifest(self) -> None:
        manifest = {
            "format": "hdf5",
            "total_samples": self._stats.total_samples,
            "total_shards": self._stats.total_shards,
            "paths": self._stats.output_paths,
        }
        path = os.path.join(self.config.output_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)


class WebDatasetWriter(BaseWriter):
    """Write processed datasets to WebDataset tar shard format.

    Each shard is a tar archive containing samples as:
      <key>.npy   - Image array (numpy format)
      <key>.json  - Metadata dictionary
      <key>.cls   - Label (text)

    This format is designed for sequential streaming reads and is
    compatible with the webdataset PyTorch library.
    """

    def __init__(self, config: WriterConfig) -> None:
        super().__init__(config)
        self._tar: Optional[tarfile.TarFile] = None
        self._shard_idx = 0
        self._shard_bytes = 0
        self._shard_samples = 0

    def open(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._start_time = time.monotonic()
        self._open_new_shard()
        logger.info("WebDatasetWriter opened: %s", self.config.output_dir)

    def _open_new_shard(self) -> None:
        if self._tar is not None:
            self._tar.close()

        path = os.path.join(
            self.config.output_dir,
            f"{self.config.shard_prefix}_{self._shard_idx:06d}.tar",
        )
        self._tar = tarfile.open(path, "w")
        self._stats.output_paths.append(path)
        self._shard_bytes = 0
        self._shard_samples = 0
        self._stats.total_shards += 1

    def write(self, sample: DatasetSample) -> None:
        if self._tar is None:
            raise RuntimeError("Writer not opened")

        # Check if current shard is full
        if (
            self._shard_bytes >= self.config.shard_size_mb * 1024 * 1024
            or self._shard_samples >= self.config.shard_max_samples
        ):
            self._shard_idx += 1
            self._open_new_shard()

        key = sample.key

        # Write image
        if self.config.image_format == "npy":
            buf = io.BytesIO()
            np.save(buf, sample.image)
            data = buf.getvalue()
            self._add_to_tar(f"{key}.npy", data)
        elif self.config.image_format == "npz":
            buf = io.BytesIO()
            np.savez_compressed(buf, image=sample.image)
            data = buf.getvalue()
            self._add_to_tar(f"{key}.npz", data)

        # Write metadata
        meta = sample.metadata or {}
        if sample.label is not None:
            meta["label"] = sample.label
        meta_bytes = json.dumps(meta).encode("utf-8")
        self._add_to_tar(f"{key}.json", meta_bytes)

        # Write label separately for easy access
        if sample.label is not None:
            label_bytes = str(sample.label).encode("utf-8")
            self._add_to_tar(f"{key}.cls", label_bytes)

        self._stats.total_samples += 1
        self._stats.total_bytes += sample.size_bytes
        self._shard_samples += 1

    def _add_to_tar(self, name: str, data: bytes) -> None:
        """Add a file to the current tar archive."""
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = int(time.time())
        self._tar.addfile(info, io.BytesIO(data))
        self._shard_bytes += len(data)

    def close(self) -> WriteStats:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

        duration = time.monotonic() - self._start_time
        self._stats.write_duration_sec = duration
        if duration > 0:
            self._stats.samples_per_second = self._stats.total_samples / duration
            self._stats.bytes_per_second = self._stats.total_bytes / duration

        if self.config.write_manifest:
            self._write_manifest()

        logger.info(
            "WebDataset write complete: %d samples in %d shards (%.1f sec)",
            self._stats.total_samples,
            self._stats.total_shards,
            self._stats.write_duration_sec,
        )
        return self._stats

    def _write_manifest(self) -> None:
        manifest = {
            "format": "webdataset",
            "total_samples": self._stats.total_samples,
            "total_shards": self._stats.total_shards,
            "shard_size_mb": self.config.shard_size_mb,
            "image_format": self.config.image_format,
            "shards": [os.path.basename(p) for p in self._stats.output_paths],
        }
        path = os.path.join(self.config.output_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)


class LMDBWriter(BaseWriter):
    """Write processed datasets to LMDB format.

    LMDB provides fast random-access reads via memory-mapped I/O.
    Each sample is stored as a key-value pair where the value is a
    msgpack-encoded dictionary containing the image and metadata.
    """

    def __init__(self, config: WriterConfig) -> None:
        super().__init__(config)
        self._env = None

    def open(self) -> None:
        import lmdb

        os.makedirs(self.config.output_dir, exist_ok=True)
        db_path = os.path.join(self.config.output_dir, "dataset.lmdb")
        self._env = lmdb.open(
            db_path,
            map_size=self.config.lmdb_map_size_gb * (1024 ** 3),
            subdir=False,
        )
        self._start_time = time.monotonic()
        self._stats.output_paths.append(db_path)
        logger.info("LMDBWriter opened: %s", db_path)

    def write(self, sample: DatasetSample) -> None:
        if self._env is None:
            raise RuntimeError("Writer not opened")

        # Serialize: image as numpy bytes + metadata as JSON
        buf = io.BytesIO()
        np.save(buf, sample.image)
        image_bytes = buf.getvalue()

        value = {
            "image": image_bytes,
            "label": sample.label,
            "metadata": json.dumps(sample.metadata or {}),
        }
        # Simple serialization: length-prefixed fields
        serialized = self._serialize_sample(value)

        with self._env.begin(write=True) as txn:
            txn.put(sample.key.encode("utf-8"), serialized)

        self._stats.total_samples += 1
        self._stats.total_bytes += sample.size_bytes

    @staticmethod
    def _serialize_sample(value: Dict[str, Any]) -> bytes:
        """Serialize sample to bytes using a simple length-prefixed format."""
        parts = []
        for key in ("image", "label", "metadata"):
            data = value.get(key, b"")
            if isinstance(data, str):
                data = data.encode("utf-8")
            elif isinstance(data, (int, float)):
                data = str(data).encode("utf-8")
            elif data is None:
                data = b""
            parts.append(struct.pack("<I", len(data)))
            parts.append(data)
        return b"".join(parts)

    def close(self) -> WriteStats:
        if self._env is not None:
            self._env.close()
            self._env = None

        self._stats.total_shards = 1
        duration = time.monotonic() - self._start_time
        self._stats.write_duration_sec = duration
        if duration > 0:
            self._stats.samples_per_second = self._stats.total_samples / duration
            self._stats.bytes_per_second = self._stats.total_bytes / duration

        if self.config.write_manifest:
            self._write_manifest()
        return self._stats

    def _write_manifest(self) -> None:
        manifest = {
            "format": "lmdb",
            "total_samples": self._stats.total_samples,
            "paths": self._stats.output_paths,
        }
        path = os.path.join(self.config.output_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)


def create_writer(config: WriterConfig) -> BaseWriter:
    """Factory function to create the appropriate writer.

    Args:
        config: Writer configuration.

    Returns:
        Configured writer instance.

    Raises:
        ValueError: If format is unknown.
    """
    writers = {
        "hdf5": HDF5Writer,
        "webdataset": WebDatasetWriter,
        "lmdb": LMDBWriter,
    }
    writer_cls = writers.get(config.format)
    if writer_cls is None:
        raise ValueError(
            f"Unknown format: {config.format}. Supported: {list(writers.keys())}"
        )
    return writer_cls(config)
