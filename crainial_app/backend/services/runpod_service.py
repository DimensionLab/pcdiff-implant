"""
Runpod Serverless GPU Service for cran-2 cranial implant inference.

Submits a defective-skull NRRD volume to a cran-2 RunPod endpoint, polls for
completion, and returns the S3 URL of the resulting implant-mask NRRD.

Endpoint contract (job_type=cran2):
    input = {
        "job_type": "cran2",
        "defective_skull": <base64-encoded NRRD bytes>,
        "input_format": "base64",
        "threshold": <float 0..1>,
        "defect_type": "bilateral" | "frontoorbital" | "parietotemporal" | "random_1" | "random_2",
            # required for v3 checkpoints (2-channel input); ignored by baseline
        "output_prefix": <string>,
    }
    output.results = {
        "implant_volume_nrrd": "<https S3 URL to implant.nrrd>",
        # legacy alias accepted: "implant_nrrd"
    }
    output.metadata = {
        "inference_time_seconds": float,
        "processing_time_seconds": float,
    }
"""

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


class RunpodError(Exception):
    """Exception raised for Runpod API errors."""

    pass


class RunpodService:
    """Service for interacting with a cran-2 RunPod serverless endpoint."""

    def __init__(
        self,
        endpoint_id: str,
        api_key: str,
        s3_bucket: str | None = None,
        s3_region: str = "eu-central-1",
    ):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        self.base_url = f"{RUNPOD_API_BASE}/{endpoint_id}"

    def _get_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def encode_nrrd_bytes(nrrd_bytes: bytes) -> str:
        return base64.b64encode(nrrd_bytes).decode("utf-8")

    @staticmethod
    def encode_nrrd_file(path: Path | str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def submit_cran2_job(
        self,
        defective_skull_nrrd: bytes | str,
        threshold: float = 0.5,
        output_prefix: str | None = None,
        defect_type: str | None = None,
    ) -> str:
        """Submit a cran-2 inference job.

        Args:
            defective_skull_nrrd: NRRD volume as raw bytes OR a base64 string.
            threshold: Binarization threshold for the predicted implant mask.
            output_prefix: S3 key prefix for results.
            defect_type: SkullBreak defect-type label. Required when the
                deployed endpoint runs a v3 checkpoint (2-channel input);
                ignored by the 1-channel baseline.

        Returns:
            RunPod job ID for polling.
        """
        if isinstance(defective_skull_nrrd, (bytes, bytearray)):
            encoded = self.encode_nrrd_bytes(bytes(defective_skull_nrrd))
        else:
            encoded = defective_skull_nrrd

        input_payload: dict = {
            "job_type": "cran2",
            "defective_skull": encoded,
            "input_format": "base64",
            "threshold": float(threshold),
            "output_prefix": output_prefix or f"cran2_{int(time.time())}",
        }
        if defect_type:
            input_payload["defect_type"] = defect_type

        payload = {
            "input": input_payload,
            # cran-2 inference is fast (~0.5s GPU); 5 minute cap covers cold starts.
            "policy": {"executionTimeout": 300000},
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/run",
                headers=self._get_headers(),
                json=payload,
            )
            if response.status_code == 401:
                raise RunpodError("Invalid API key - check your Runpod API key")
            if response.status_code == 404:
                raise RunpodError(f"Endpoint not found: {self.endpoint_id}")
            if response.status_code == 429:
                raise RunpodError("Rate limited - too many requests")
            if response.status_code != 200:
                raise RunpodError(f"Failed to submit job: {response.status_code} - {response.text}")

            result = response.json()
            job_id = result.get("id")
            if not job_id:
                raise RunpodError(f"No job ID in response: {result}")

            logger.info(f"Submitted cran-2 job {job_id} (status={result.get('status')})")
            return job_id

    async def get_job_status(self, job_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/status/{job_id}",
                headers=self._get_headers(),
            )
            if response.status_code == 404:
                raise RunpodError(f"Job not found: {job_id}")
            if response.status_code != 200:
                raise RunpodError(f"Failed to get job status: {response.status_code} - {response.text}")
            return response.json()

    async def wait_for_completion(
        self,
        job_id: str,
        progress_callback: Callable[[str, int], None] | None = None,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
        should_cancel_callback: Callable[[], bool] | None = None,
    ) -> dict:
        start = time.time()
        last_status = None

        while True:
            if time.time() - start > timeout:
                raise RunpodError(f"Job {job_id} timed out after {timeout}s")

            if should_cancel_callback and should_cancel_callback():
                logger.info(f"Cancellation requested for job {job_id}")
                await self.cancel_job(job_id)
                raise RunpodError(f"Job {job_id} was cancelled by user")

            try:
                status_response = await self.get_job_status(job_id)
            except RunpodError as e:
                logger.warning(f"Error polling job status: {e}")
                await asyncio.sleep(poll_interval)
                continue

            status = status_response.get("status", "UNKNOWN")
            progress_map = {
                "IN_QUEUE": 10,
                "IN_PROGRESS": 50,
                "COMPLETED": 100,
                "FAILED": 100,
                "CANCELLED": 100,
            }
            progress = progress_map.get(status, 5)

            if status != last_status:
                logger.info(f"Job {job_id} status: {status}")
                last_status = status
            if progress_callback:
                progress_callback(status, progress)

            if status == "COMPLETED":
                return status_response
            if status == "FAILED":
                raise RunpodError(f"Job {job_id} failed: {status_response.get('error', 'Unknown error')}")
            if status == "CANCELLED":
                raise RunpodError(f"Job {job_id} was cancelled")

            await asyncio.sleep(poll_interval)

    async def cancel_job(self, job_id: str) -> bool:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/cancel/{job_id}",
                headers=self._get_headers(),
            )
            if response.status_code == 200:
                logger.info(f"Cancelled job {job_id}: {response.json().get('status')}")
                return True
            logger.warning(f"Failed to cancel job {job_id}: {response.status_code}")
            return False

    async def get_health(self) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/health",
                headers=self._get_headers(),
            )
            if response.status_code != 200:
                raise RunpodError(f"Failed to get health: {response.status_code} - {response.text}")
            return response.json()

    # ------------------------------------------------------------------
    # Synchronous wrappers (used from background task threads)
    # ------------------------------------------------------------------

    def _run_async(self, coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def submit_cran2_job_sync(
        self,
        defective_skull_nrrd: bytes | str,
        threshold: float = 0.5,
        output_prefix: str | None = None,
        defect_type: str | None = None,
    ) -> str:
        return self._run_async(
            self.submit_cran2_job(defective_skull_nrrd, threshold, output_prefix, defect_type)
        )

    def get_job_status_sync(self, job_id: str) -> dict:
        return self._run_async(self.get_job_status(job_id))

    def wait_for_completion_sync(
        self,
        job_id: str,
        progress_callback: Callable[[str, int], None] | None = None,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
        should_cancel_callback: Callable[[], bool] | None = None,
    ) -> dict:
        return self._run_async(
            self.wait_for_completion(job_id, progress_callback, poll_interval, timeout, should_cancel_callback)
        )

    def cancel_job_sync(self, job_id: str) -> bool:
        return self._run_async(self.cancel_job(job_id))

    def get_health_sync(self) -> dict:
        return self._run_async(self.get_health())


def download_from_s3_url(
    s3_url: str,
    local_path: Path,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    s3_region: str | None = None,
) -> Path:
    """Download a file from an S3 HTTPS URL to local_path.

    Uses boto3 with AWS credentials if available, otherwise falls back to a
    plain HTTPS GET (only works for public objects).
    """
    import os
    import re

    local_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(s3_url)
    host = parsed.netloc
    path = parsed.path.lstrip("/")
    bucket = key = region = None

    match = re.match(r"^([^.]+)\.s3\.([^.]+)\.amazonaws\.com$", host)
    if match:
        bucket, region = match.group(1), match.group(2)
        key = path
    else:
        match = re.match(r"^s3\.([^.]+)\.amazonaws\.com$", host)
        if match:
            region = match.group(1)
            parts = path.split("/", 1)
            if len(parts) == 2:
                bucket, key = parts

    if bucket and key:
        try:
            import boto3
            from botocore.config import Config

            access_key = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
            secret_key = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
            effective_region = s3_region or region

            if access_key and secret_key:
                s3 = boto3.client(
                    "s3",
                    region_name=effective_region,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    config=Config(signature_version="s3v4"),
                )
                logger.info(f"Downloading s3://{bucket}/{key} via boto3...")
                s3.download_file(bucket, key, str(local_path))
                logger.info(f"Downloaded {s3_url} to {local_path}")
                return local_path
            logger.warning("No AWS credentials found, falling back to HTTP GET")
        except ImportError:
            logger.warning("boto3 not installed, falling back to HTTP GET")
        except Exception as e:
            logger.warning(f"boto3 download failed: {e}, falling back to HTTP GET")

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(s3_url)
        response.raise_for_status()
        local_path.write_bytes(response.content)

    logger.info(f"Downloaded {s3_url} to {local_path}")
    return local_path


def parse_runpod_results(output: dict) -> dict:
    """Parse a completed cran-2 job's output into a structured result."""
    if output.get("status") == "error":
        raise RunpodError(f"Job failed: {output.get('error', 'Unknown error')}")

    results = output.get("results", {}) or {}
    metadata = output.get("metadata", {}) or {}

    nrrd_url = results.get("implant_volume_nrrd") or results.get("implant_nrrd")

    return {
        "s3_urls": results,
        "implant_nrrd_url": nrrd_url,
        "inference_time_seconds": metadata.get("inference_time_seconds"),
        "processing_time_seconds": metadata.get("processing_time_seconds"),
        "model_source": metadata.get("model_source"),
    }
