"""
Runpod Serverless GPU Service for cloud-based implant generation.

This service handles communication with the Runpod serverless endpoint
for running PCDiff + Voxelization inference on cloud GPUs.

The service supports:
- Submitting generation jobs to Runpod (/run for async, /runsync for sync)
- Polling for job status (/status/{job_id})
- Cancelling jobs (/cancel/{job_id})
- Downloading results from S3
- Caching results locally

Runpod API Reference:
- /run: Submit async job, returns job ID immediately
- /runsync: Submit sync job, wait for completion (max 5 min with ?wait=300000)
- /status/{job_id}: Get job status and results
- /cancel/{job_id}: Cancel a running job
- /health: Check endpoint health

Job statuses: IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
"""

import asyncio
import base64
import io
import logging
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Runpod API endpoints
RUNPOD_API_BASE = "https://api.runpod.ai/v2"


class RunpodError(Exception):
    """Exception raised for Runpod API errors."""
    pass


class RunpodService:
    """
    Service for interacting with Runpod serverless GPU endpoints.
    
    Usage:
        service = RunpodService(endpoint_id="6on3tc0nzlyt42", api_key="your_key")
        
        # Async usage
        job_id = await service.submit_job(points, num_ensemble=1)
        result = await service.wait_for_completion(job_id)
        
        # Sync usage (for background tasks)
        job_id = service.submit_job_sync(points, num_ensemble=1)
        result = service.wait_for_completion_sync(job_id)
    """

    def __init__(
        self,
        endpoint_id: str,
        api_key: str,
        s3_bucket: str | None = None,
        s3_region: str = "us-east-1",
    ):
        """
        Initialize the Runpod service.

        Args:
            endpoint_id: The Runpod endpoint ID (e.g., "6on3tc0nzlyt42")
            api_key: Runpod API key for authentication
            s3_bucket: Optional S3 bucket name for result storage
            s3_region: AWS region for S3 bucket
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        self.base_url = f"{RUNPOD_API_BASE}/{endpoint_id}"

    def _get_headers(self) -> dict:
        """Get HTTP headers for Runpod API requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _encode_point_cloud(self, points: np.ndarray) -> str:
        """Encode numpy point cloud as base64 string."""
        buffer = io.BytesIO()
        np.save(buffer, points.astype(np.float32))
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    async def submit_job(
        self,
        defective_skull_points: np.ndarray,
        num_ensemble: int = 1,
        sampling_steps: int = 1000,
        output_prefix: str | None = None,
        pcdiff_model: str = "best",
        voxelization_resolution: int = 512,
        smoothing_iterations: int = 0,
        close_holes: bool = False,
    ) -> str:
        """
        Submit an async generation job to Runpod using /run endpoint.

        Args:
            defective_skull_points: Input point cloud as numpy array (N, 3)
            num_ensemble: Number of ensemble samples to generate
            sampling_steps: Number of diffusion steps
            output_prefix: Optional prefix for S3 output keys
            pcdiff_model: Which PCDiff model to use ("best" or "latest")
            voxelization_resolution: PSR grid resolution (128, 256, 512, 1024)
            smoothing_iterations: Laplacian smoothing iterations (0 = disabled)
            close_holes: Whether to fill holes in the generated mesh

        Returns:
            Job ID for tracking the request
        """
        encoded_data = self._encode_point_cloud(defective_skull_points)

        payload = {
            "input": {
                "job_type": "full",
                "defective_skull": encoded_data,
                "input_format": "base64",
                "num_ensemble": num_ensemble,
                "sampling_steps": sampling_steps,
                "voxelization_resolution": voxelization_resolution,
                "smoothing_iterations": smoothing_iterations,
                "close_holes": close_holes,
                "output_prefix": output_prefix or f"job_{int(time.time())}",
                "pcdiff_model": pcdiff_model,
            },
            # Set execution timeout to 60 minutes (3600000ms) for DDPM with 1000 steps
            "policy": {
                "executionTimeout": 3600000,
            }
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
                raise RunpodError(
                    f"Failed to submit job: {response.status_code} - {response.text}"
                )

            result = response.json()
            job_id = result.get("id")
            status = result.get("status")

            if not job_id:
                raise RunpodError(f"No job ID in response: {result}")

            logger.info(f"Submitted Runpod job: {job_id} (status: {status})")
            return job_id

    async def submit_revoxelization_job(
        self,
        implant_points: np.ndarray,
        defective_skull_points: np.ndarray,
        voxelization_resolution: int = 512,
        output_prefix: str | None = None,
        smoothing_iterations: int = 0,
        close_holes: bool = False,
    ) -> str:
        """
        Submit a re-voxelization job to Runpod (mesh generation only, no diffusion).

        Args:
            implant_points: Existing implant point cloud (N, 3)
            defective_skull_points: Defective skull point cloud (M, 3)
            voxelization_resolution: PSR grid resolution (128, 256, 512, 1024)
            output_prefix: Optional prefix for S3 output keys
            smoothing_iterations: Laplacian smoothing iterations (0 = disabled)
            close_holes: Whether to fill holes in the generated mesh

        Returns:
            Job ID for tracking the request
        """
        encoded_implant = self._encode_point_cloud(implant_points)
        encoded_defective = self._encode_point_cloud(defective_skull_points)

        payload = {
            "input": {
                "job_type": "revoxelize",
                "implant_points": encoded_implant,
                "defective_skull": encoded_defective,
                "input_format": "base64",
                "voxelization_resolution": voxelization_resolution,
                "smoothing_iterations": smoothing_iterations,
                "close_holes": close_holes,
                "output_prefix": output_prefix or f"revox_{int(time.time())}",
            },
            # Re-voxelization is fast, 5 minute timeout should be plenty
            "policy": {
                "executionTimeout": 300000,
            }
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
                raise RunpodError(
                    f"Failed to submit revox job: {response.status_code} - {response.text}"
                )

            result = response.json()
            job_id = result.get("id")
            status = result.get("status")

            if not job_id:
                raise RunpodError(f"No job ID in response: {result}")

            logger.info(f"Submitted Runpod re-voxelization job: {job_id} (status: {status}, resolution: {voxelization_resolution})")
            return job_id

    async def get_job_status(self, job_id: str) -> dict:
        """
        Get the status of a Runpod job using /status/{job_id} endpoint.

        Args:
            job_id: The Runpod job ID

        Returns:
            Job status dictionary with keys:
            - status: IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
            - output: Job output (if COMPLETED)
            - error: Error message (if FAILED)
            - delayTime: Time spent in queue (ms)
            - executionTime: Time spent processing (ms)
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/status/{job_id}",
                headers=self._get_headers(),
            )

            if response.status_code == 404:
                raise RunpodError(f"Job not found: {job_id}")
            if response.status_code != 200:
                raise RunpodError(
                    f"Failed to get job status: {response.status_code} - {response.text}"
                )

            return response.json()

    async def wait_for_completion(
        self,
        job_id: str,
        progress_callback: Callable[[str, int], None] | None = None,
        poll_interval: float = 3.0,
        timeout: float = 1200.0,
        should_cancel_callback: Callable[[], bool] | None = None,
    ) -> dict:
        """
        Wait for a Runpod job to complete, polling for status.

        Args:
            job_id: The Runpod job ID
            progress_callback: Optional callback(status, progress_percent)
            poll_interval: Seconds between status checks (default 3s)
            timeout: Maximum seconds to wait (default 20 min)
            should_cancel_callback: Optional callback that returns True if job should be cancelled

        Returns:
            Final job result dictionary with output
        """
        start_time = time.time()
        last_status = None

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise RunpodError(f"Job {job_id} timed out after {timeout}s")
            
            # Check if we should cancel
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

            job_status = status_response.get("status", "UNKNOWN")

            # Map Runpod status to progress percentage
            progress_map = {
                "IN_QUEUE": 10,
                "IN_PROGRESS": 50,
                "COMPLETED": 100,
                "FAILED": 100,
                "CANCELLED": 100,
            }
            progress = progress_map.get(job_status, 5)

            # Only log/callback on status change
            if job_status != last_status:
                logger.info(f"Job {job_id} status: {job_status}")
                last_status = job_status

            if progress_callback:
                progress_callback(job_status, progress)

            if job_status == "COMPLETED":
                return status_response

            if job_status == "FAILED":
                error = status_response.get("error", "Unknown error")
                raise RunpodError(f"Job {job_id} failed: {error}")

            if job_status == "CANCELLED":
                raise RunpodError(f"Job {job_id} was cancelled")

            await asyncio.sleep(poll_interval)

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running Runpod job using /cancel/{job_id} endpoint.

        Args:
            job_id: The Runpod job ID

        Returns:
            True if cancellation was successful
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/cancel/{job_id}",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Cancelled job {job_id}: {result.get('status')}")
                return True
            else:
                logger.warning(
                    f"Failed to cancel job {job_id}: {response.status_code}"
                )
                return False

    async def get_health(self) -> dict:
        """
        Get endpoint health status using /health endpoint.

        Returns:
            Health status with worker and job statistics
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/health",
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise RunpodError(
                    f"Failed to get health: {response.status_code} - {response.text}"
                )

            return response.json()

    # -------------------------------------------------------------------------
    # Synchronous wrappers for use in background tasks
    # -------------------------------------------------------------------------

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)

    def submit_job_sync(
        self,
        defective_skull_points: np.ndarray,
        num_ensemble: int = 1,
        sampling_steps: int = 1000,
        output_prefix: str | None = None,
        pcdiff_model: str = "best",
        voxelization_resolution: int = 512,
        smoothing_iterations: int = 0,
        close_holes: bool = False,
    ) -> str:
        """Synchronous wrapper for submit_job."""
        return self._run_async(
            self.submit_job(
                defective_skull_points,
                num_ensemble,
                sampling_steps,
                output_prefix,
                pcdiff_model,
                voxelization_resolution,
                smoothing_iterations,
                close_holes,
            )
        )

    def get_job_status_sync(self, job_id: str) -> dict:
        """Synchronous wrapper for get_job_status."""
        return self._run_async(self.get_job_status(job_id))

    def wait_for_completion_sync(
        self,
        job_id: str,
        progress_callback: Callable[[str, int], None] | None = None,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
        should_cancel_callback: Callable[[], bool] | None = None,
    ) -> dict:
        """Synchronous wrapper for wait_for_completion."""
        return self._run_async(
            self.wait_for_completion(job_id, progress_callback, poll_interval, timeout, should_cancel_callback)
        )

    def cancel_job_sync(self, job_id: str) -> bool:
        """Synchronous wrapper for cancel_job."""
        return self._run_async(self.cancel_job(job_id))

    def get_health_sync(self) -> dict:
        """Synchronous wrapper for get_health."""
        return self._run_async(self.get_health())


def download_from_s3_url(
    s3_url: str, 
    local_path: Path,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> Path:
    """
    Download a file from S3 HTTPS URL to local path.
    
    Uses boto3 with credentials if provided, otherwise falls back to
    environment variables or public HTTP GET.

    Args:
        s3_url: Full S3 HTTPS URL (e.g., https://bucket.s3.region.amazonaws.com/key)
        local_path: Local path to save the file
        aws_access_key_id: Optional AWS access key (uses env var if not provided)
        aws_secret_access_key: Optional AWS secret key (uses env var if not provided)

    Returns:
        Path to the downloaded file
    """
    import os
    import re
    
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse S3 URL to extract bucket and key
    # Format: https://bucket.s3.region.amazonaws.com/key
    # or: https://s3.region.amazonaws.com/bucket/key
    parsed = urlparse(s3_url)
    host = parsed.netloc
    path = parsed.path.lstrip('/')
    
    # Try to extract bucket and key from URL
    bucket = None
    key = None
    region = None
    
    # Pattern 1: bucket.s3.region.amazonaws.com/key
    match = re.match(r'^([^.]+)\.s3\.([^.]+)\.amazonaws\.com$', host)
    if match:
        bucket = match.group(1)
        region = match.group(2)
        key = path
    else:
        # Pattern 2: s3.region.amazonaws.com/bucket/key
        match = re.match(r'^s3\.([^.]+)\.amazonaws\.com$', host)
        if match:
            region = match.group(1)
            parts = path.split('/', 1)
            if len(parts) == 2:
                bucket, key = parts
    
    if bucket and key:
        # Use boto3 for authenticated download
        try:
            import boto3
            from botocore.config import Config
            
            # Get credentials from params, env vars, or AWS config
            access_key = aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
            secret_key = aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
            
            if access_key and secret_key:
                s3_client = boto3.client(
                    's3',
                    region_name=region,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    config=Config(signature_version='s3v4'),
                )
                
                logger.info(f"Downloading s3://{bucket}/{key} using boto3...")
                s3_client.download_file(bucket, key, str(local_path))
                logger.info(f"Downloaded {s3_url} to {local_path}")
                return local_path
            else:
                logger.warning("No AWS credentials found, falling back to HTTP GET")
        except ImportError:
            logger.warning("boto3 not installed, falling back to HTTP GET")
        except Exception as e:
            logger.warning(f"boto3 download failed: {e}, falling back to HTTP GET")
    
    # Fallback: simple HTTP GET (only works for public buckets)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(s3_url)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(response.content)

    logger.info(f"Downloaded {s3_url} to {local_path}")
    return local_path


def parse_runpod_results(output: dict) -> dict:
    """
    Parse Runpod job output into a structured result.

    Args:
        output: The 'output' field from a completed Runpod job

    Returns:
        Dictionary with parsed results including S3 URLs
    """
    if output.get("status") == "error":
        raise RunpodError(f"Job failed: {output.get('error', 'Unknown error')}")

    results = output.get("results", {})
    metadata = output.get("metadata", {})

    return {
        "s3_urls": results,
        "processing_time_seconds": metadata.get("processing_time_seconds"),
        "num_implant_points": metadata.get("num_implant_points"),
        "mesh_vertices": metadata.get("mesh_vertices"),
        "mesh_faces": metadata.get("mesh_faces"),
        "model_source": metadata.get("model_source"),
    }
