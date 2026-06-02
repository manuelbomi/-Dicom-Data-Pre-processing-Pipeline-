"""AWS S3 integration for cloud-scale DICOM processing.

Provides streaming DICOM reads from S3, parallel uploads of processed datasets,
presigned URL generation, and SageMaker Training data channel compatibility.

Designed for large-scale workflows where staging terabytes of DICOM data to
local disk is impractical. Key features:

  - Streaming reads: Process DICOM files directly from S3 without full download.
  - Parallel transfers: Configurable concurrency for both reads and writes.
  - Multipart uploads: Automatic chunked upload for large shard files.
  - SageMaker integration: Output manifests compatible with SageMaker Training
    input data channels (S3DataSource with ManifestFile).
  - Presigned URLs: Generate time-limited access URLs for dataset sharing.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    """Configuration for S3 operations."""

    bucket: str = ""
    prefix: str = ""
    region: str = "us-east-1"
    max_concurrent_requests: int = 32
    multipart_threshold_mb: int = 64
    multipart_chunksize_mb: int = 16
    max_retries: int = 3
    connect_timeout_sec: int = 10
    read_timeout_sec: int = 60
    use_accelerate: bool = False
    server_side_encryption: str = ""  # "AES256", "aws:kms", or ""
    kms_key_id: str = ""
    storage_class: str = "STANDARD"  # "STANDARD", "INTELLIGENT_TIERING", etc.


@dataclass
class TransferStats:
    """Statistics for S3 transfer operations."""

    files_transferred: int = 0
    bytes_transferred: int = 0
    failed_transfers: int = 0
    duration_sec: float = 0.0
    errors: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def throughput_mbps(self) -> float:
        if self.duration_sec <= 0:
            return 0.0
        return (self.bytes_transferred / 1024 / 1024) / self.duration_sec


class S3Client:
    """Wrapper around boto3 S3 client with optimized transfer configuration.

    Configures boto3 with appropriate timeouts, retry logic, and transfer
    settings for large-scale medical imaging workloads.

    Args:
        config: S3 configuration.
    """

    def __init__(self, config: Optional[S3Config] = None) -> None:
        self.config = config or S3Config()
        self._client = None
        self._transfer_config = None

    @property
    def client(self) -> Any:
        """Lazy-initialize the boto3 S3 client."""
        if self._client is None:
            import boto3
            from botocore.config import Config as BotoConfig

            boto_config = BotoConfig(
                region_name=self.config.region,
                retries={"max_attempts": self.config.max_retries, "mode": "adaptive"},
                connect_timeout=self.config.connect_timeout_sec,
                read_timeout=self.config.read_timeout_sec,
                max_pool_connections=self.config.max_concurrent_requests,
                s3={"use_accelerate_endpoint": self.config.use_accelerate},
            )
            self._client = boto3.client("s3", config=boto_config)
        return self._client

    @property
    def transfer_config(self) -> Any:
        """Lazy-initialize the S3 transfer configuration."""
        if self._transfer_config is None:
            from boto3.s3.transfer import TransferConfig

            self._transfer_config = TransferConfig(
                multipart_threshold=self.config.multipart_threshold_mb * 1024 * 1024,
                multipart_chunksize=self.config.multipart_chunksize_mb * 1024 * 1024,
                max_concurrency=self.config.max_concurrent_requests,
                use_threads=True,
            )
        return self._transfer_config


class S3DicomStore:
    """S3-backed DICOM file store with streaming read capability.

    Enables processing of DICOM archives stored in S3 without staging
    the full dataset to local disk. Supports listing, streaming reads,
    and parallel iteration.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix for DICOM files.
        region: AWS region.
        max_concurrent: Maximum parallel S3 requests.

    Example::

        store = S3DicomStore(
            bucket="clinical-imaging-archive",
            prefix="screening-mammography/2023/",
        )
        for dicom_bytes in store.iter_dicom_files(max_concurrent=64):
            ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
            # process...
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        max_concurrent: int = 32,
    ) -> None:
        config = S3Config(
            bucket=bucket,
            prefix=prefix,
            region=region,
            max_concurrent_requests=max_concurrent,
        )
        self._s3 = S3Client(config)
        self.bucket = bucket
        self.prefix = prefix
        self.max_concurrent = max_concurrent

    def list_dicom_keys(
        self,
        extensions: Optional[List[str]] = None,
        max_keys: Optional[int] = None,
    ) -> List[str]:
        """List all DICOM file keys under the configured prefix.

        Uses S3 pagination to handle buckets with millions of objects.

        Args:
            extensions: File extensions to filter (e.g., [".dcm", ""]).
            max_keys: Maximum number of keys to return.

        Returns:
            List of S3 object keys.
        """
        if extensions is None:
            extensions = [".dcm", ".dicom", ".DCM", ""]

        keys: List[str] = []
        paginator = self._s3.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = os.path.splitext(key)[1]
                if ext in extensions or not extensions:
                    keys.append(key)
                    if max_keys and len(keys) >= max_keys:
                        return keys

        logger.info("Listed %d DICOM keys under s3://%s/%s", len(keys), self.bucket, self.prefix)
        return keys

    def read_dicom_bytes(self, key: str) -> bytes:
        """Read a single DICOM file from S3 into memory.

        Args:
            key: S3 object key.

        Returns:
            Raw file bytes.
        """
        response = self._s3.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def stream_dicom(self, key: str, chunk_size: int = 8192) -> Generator[bytes, None, None]:
        """Stream a DICOM file from S3 in chunks.

        For very large files (e.g., whole-slide images or tomosynthesis),
        this avoids loading the entire file into memory.

        Args:
            key: S3 object key.
            chunk_size: Read chunk size in bytes.

        Yields:
            Byte chunks.
        """
        response = self._s3.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def iter_dicom_files(
        self,
        max_concurrent: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> Generator[Tuple[str, bytes], None, None]:
        """Iterate over DICOM files with parallel prefetching.

        Uses a thread pool to prefetch files from S3 while the main
        thread processes the current file.

        Args:
            max_concurrent: Override concurrent request count.
            max_files: Maximum files to iterate.

        Yields:
            Tuples of (key, file_bytes).
        """
        keys = self.list_dicom_keys(max_keys=max_files)
        concurrency = max_concurrent or self.max_concurrent

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {}
            key_iter = iter(keys)

            # Seed the pool
            for _ in range(min(concurrency, len(keys))):
                try:
                    key = next(key_iter)
                    futures[pool.submit(self.read_dicom_bytes, key)] = key
                except StopIteration:
                    break

            while futures:
                done_futures = [f for f in futures if f.done()]
                if not done_futures:
                    # Wait for at least one
                    done = next(as_completed(futures))
                    done_futures = [done]

                for future in done_futures:
                    key = futures.pop(future)
                    try:
                        data = future.result()
                        yield key, data
                    except Exception as exc:
                        logger.warning("Failed to read %s: %s", key, exc)

                    # Submit next
                    try:
                        next_key = next(key_iter)
                        futures[pool.submit(self.read_dicom_bytes, next_key)] = next_key
                    except StopIteration:
                        pass


class S3DatasetUploader:
    """Upload processed datasets to S3 with parallel transfers.

    Handles uploading shard files, manifests, and generating SageMaker-
    compatible data channel configurations.

    Args:
        config: S3 configuration.
    """

    def __init__(self, config: S3Config) -> None:
        self._s3 = S3Client(config)
        self.config = config

    def upload_directory(
        self,
        local_dir: str,
        s3_prefix: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
    ) -> TransferStats:
        """Upload all files in a local directory to S3.

        Args:
            local_dir: Local directory path.
            s3_prefix: S3 key prefix. Defaults to config prefix.
            include_patterns: File patterns to include (e.g., ["*.tar", "*.json"]).

        Returns:
            Transfer statistics.
        """
        prefix = s3_prefix or self.config.prefix
        stats = TransferStats()
        t0 = time.monotonic()

        # Collect files to upload
        files_to_upload: List[Tuple[str, str]] = []
        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                if include_patterns:
                    if not any(fname.endswith(p.lstrip("*")) for p in include_patterns):
                        continue
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{prefix.rstrip('/')}/{rel_path}".replace("\\", "/")
                files_to_upload.append((local_path, s3_key))

        logger.info("Uploading %d files to s3://%s/%s", len(files_to_upload), self.config.bucket, prefix)

        # Parallel upload
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as pool:
            futures = {
                pool.submit(self._upload_file, local, key): (local, key)
                for local, key in files_to_upload
            }

            for future in as_completed(futures):
                local_path, s3_key = futures[future]
                try:
                    nbytes = future.result()
                    stats.files_transferred += 1
                    stats.bytes_transferred += nbytes
                except Exception as exc:
                    stats.failed_transfers += 1
                    stats.errors.append((local_path, str(exc)))
                    logger.error("Upload failed for %s: %s", local_path, exc)

        stats.duration_sec = time.monotonic() - t0
        logger.info(
            "Upload complete: %d files, %.1f MB, %.1f MB/s",
            stats.files_transferred,
            stats.bytes_transferred / 1024 / 1024,
            stats.throughput_mbps,
        )
        return stats

    def _upload_file(self, local_path: str, s3_key: str) -> int:
        """Upload a single file to S3."""
        extra_args: Dict[str, str] = {}
        if self.config.server_side_encryption:
            extra_args["ServerSideEncryption"] = self.config.server_side_encryption
        if self.config.kms_key_id:
            extra_args["SSEKMSKeyId"] = self.config.kms_key_id
        if self.config.storage_class:
            extra_args["StorageClass"] = self.config.storage_class

        file_size = os.path.getsize(local_path)
        self._s3.client.upload_file(
            local_path,
            self.config.bucket,
            s3_key,
            Config=self._s3.transfer_config,
            ExtraArgs=extra_args if extra_args else None,
        )
        return file_size

    def generate_presigned_urls(
        self,
        keys: List[str],
        expiration_sec: int = 3600,
    ) -> List[str]:
        """Generate presigned URLs for dataset files.

        Useful for sharing processed datasets with collaborators or
        external annotation services without granting bucket-level access.

        Args:
            keys: S3 object keys.
            expiration_sec: URL expiration time in seconds.

        Returns:
            List of presigned URLs.
        """
        urls = []
        for key in keys:
            url = self._s3.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket, "Key": key},
                ExpiresIn=expiration_sec,
            )
            urls.append(url)
        return urls

    def generate_sagemaker_manifest(
        self,
        s3_prefix: str,
        output_key: str,
        content_type: str = "application/x-npy",
    ) -> str:
        """Generate a SageMaker-compatible manifest file.

        Creates a manifest.json that can be used as a SageMaker Training
        input data channel with S3DataDistributionType=ShardedByS3Key.

        Args:
            s3_prefix: S3 prefix containing the dataset shards.
            output_key: S3 key for the manifest file.
            content_type: MIME type of the data files.

        Returns:
            S3 URI of the uploaded manifest.
        """
        # List shard files
        keys = []
        paginator = self._s3.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith((".tar", ".h5", ".lmdb")):
                    keys.append(obj["Key"])

        # Create manifest in SageMaker format
        manifest_lines = []
        for key in sorted(keys):
            entry = {
                "prefix": f"s3://{self.config.bucket}/{key}",
            }
            manifest_lines.append(json.dumps(entry))

        manifest_content = "\n".join(manifest_lines)
        self._s3.client.put_object(
            Bucket=self.config.bucket,
            Key=output_key,
            Body=manifest_content.encode("utf-8"),
            ContentType="application/json",
        )

        manifest_uri = f"s3://{self.config.bucket}/{output_key}"
        logger.info("SageMaker manifest written: %s (%d entries)", manifest_uri, len(keys))
        return manifest_uri
