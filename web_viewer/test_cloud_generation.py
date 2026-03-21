#!/usr/bin/env python3
"""
Test script for cloud generation feature.

This script tests the Runpod cloud generation integration by:
1. Checking settings API
2. Configuring cloud settings
3. Creating a test project and uploading a defective skull
4. Submitting a cloud generation job
5. Polling for completion
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# API base URL
API_BASE = "http://localhost:8080"

# Test data
TEST_DATA_PATH = Path(__file__).parent.parent / "datasets/SkullBreak/defective_skull/bilateral/000_surf.npy"


def test_settings_api():
    """Test that settings API works and returns cloud settings."""
    print("\n=== Testing Settings API ===")
    
    response = httpx.get(f"{API_BASE}/api/v1/settings/")
    print(f"Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return False
    
    settings = response.json()
    print(f"Settings: {json.dumps(settings, indent=2)}")
    
    # Check cloud settings exist
    assert "cloud_generation_enabled" in settings, "Missing cloud_generation_enabled"
    assert "runpod_endpoint_id" in settings, "Missing runpod_endpoint_id"
    assert "runpod_api_key_set" in settings, "Missing runpod_api_key_set"
    
    print("✓ Settings API works correctly")
    return True


def test_system_info():
    """Test system info endpoint."""
    print("\n=== Testing System Info ===")
    
    response = httpx.get(f"{API_BASE}/api/v1/settings/system-info")
    print(f"Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return False
    
    info = response.json()
    print(f"System Info: {json.dumps(info, indent=2)}")
    
    # Check cloud_configured field
    assert "cloud_configured" in info, "Missing cloud_configured"
    print(f"Cloud configured: {info['cloud_configured']}")
    
    print("✓ System info API works correctly")
    return True


def configure_cloud_settings():
    """Configure cloud settings for testing."""
    print("\n=== Configuring Cloud Settings ===")
    
    # Get API key from environment
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("RUNPOD_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    
    if not api_key:
        print("Warning: No RUNPOD_API_KEY found")
        return False
    
    print(f"API Key found: {api_key[:10]}...")
    
    # Update settings
    update_data = {
        "cloud_generation_enabled": True,
        "runpod_endpoint_id": "6on3tc0nzlyt42",
        "runpod_api_key": api_key,
    }
    
    response = httpx.put(
        f"{API_BASE}/api/v1/settings/",
        json=update_data,
    )
    
    print(f"Update status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return False
    
    settings = response.json()
    print(f"Updated settings: cloud_generation_enabled={settings['cloud_generation_enabled']}, runpod_api_key_set={settings['runpod_api_key_set']}")
    
    print("✓ Cloud settings configured")
    return True


def get_or_create_project():
    """Get or create a test project."""
    print("\n=== Getting/Creating Test Project ===")
    
    # List projects
    response = httpx.get(f"{API_BASE}/api/v1/projects/")
    if response.status_code != 200:
        print(f"Error listing projects: {response.text}")
        return None
    
    projects = response.json()
    
    # Look for existing test project
    for project in projects:
        if project["name"] == "Cloud Test Project":
            print(f"Found existing project: {project['id']}")
            return project["id"]
    
    # Create new project
    response = httpx.post(
        f"{API_BASE}/api/v1/projects/",
        json={"name": "Cloud Test Project", "description": "Testing cloud generation"},
    )
    
    if response.status_code != 201:
        print(f"Error creating project: {response.text}")
        return None
    
    project = response.json()
    print(f"Created project: {project['id']}")
    return project["id"]


def get_or_upload_point_cloud(project_id: str):
    """Get or upload a test point cloud."""
    print("\n=== Getting/Uploading Test Point Cloud ===")
    
    # List point clouds in project
    response = httpx.get(f"{API_BASE}/api/v1/point-clouds/", params={"project_id": project_id})
    if response.status_code != 200:
        print(f"Error listing point clouds: {response.text}")
        return None
    
    point_clouds = response.json()
    
    # Look for existing defective skull
    for pc in point_clouds:
        if pc["scan_category"] == "defective_skull" and pc["file_format"] == "npy":
            print(f"Found existing point cloud: {pc['id']} ({pc['name']})")
            return pc["id"]
    
    # Upload new point cloud
    if not TEST_DATA_PATH.exists():
        print(f"Test data not found: {TEST_DATA_PATH}")
        return None
    
    # Load and verify point cloud
    points = np.load(str(TEST_DATA_PATH))
    print(f"Loaded point cloud: shape={points.shape}, dtype={points.dtype}")
    
    # Register point cloud via API
    response = httpx.post(
        f"{API_BASE}/api/v1/point-clouds/",
        json={
            "name": "Test Defective Skull 000",
            "description": "Bilateral defect for cloud generation test",
            "file_path": str(TEST_DATA_PATH),
            "file_format": "npy",
            "num_points": points.shape[0],
            "point_dims": points.shape[1] if len(points.shape) > 1 else 3,
            "scan_category": "defective_skull",
            "defect_type": "bilateral",
            "project_id": project_id,
        },
    )
    
    if response.status_code != 201:
        print(f"Error uploading point cloud: {response.text}")
        return None
    
    pc = response.json()
    print(f"Registered point cloud: {pc['id']}")
    return pc["id"]


def submit_cloud_generation_job(project_id: str, input_pc_id: str):
    """Submit a cloud generation job."""
    print("\n=== Submitting Cloud Generation Job ===")
    
    job_data = {
        "project_id": project_id,
        "input_pc_id": input_pc_id,
        "sampling_method": "ddim",
        "sampling_steps": 50,
        "num_ensemble": 1,
        "name": "Cloud Test Generation",
        "use_cloud": True,
    }
    
    print(f"Job data: {json.dumps(job_data, indent=2)}")
    
    response = httpx.post(
        f"{API_BASE}/api/v1/generation-jobs/",
        json=job_data,
        timeout=30.0,
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code != 201:
        print(f"Error: {response.text}")
        return None
    
    job = response.json()
    print(f"Created job: {job['id']} (status: {job['status']})")
    return job["id"]


def poll_job_status(job_id: str, timeout: float = 600.0):
    """Poll job status until completion."""
    print("\n=== Polling Job Status ===")
    
    start_time = time.time()
    last_status = None
    last_step = None
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"Timeout after {timeout}s")
            return None
        
        response = httpx.get(f"{API_BASE}/api/v1/generation-jobs/{job_id}")
        if response.status_code != 200:
            print(f"Error getting job: {response.text}")
            time.sleep(2)
            continue
        
        job = response.json()
        status = job["status"]
        step = job.get("current_step", "")
        progress = job.get("progress_percent", 0)
        
        # Only print on change
        if status != last_status or step != last_step:
            print(f"[{elapsed:.0f}s] Status: {status}, Progress: {progress}%, Step: {step}")
            last_status = status
            last_step = step
        
        if status == "completed":
            print(f"\n✓ Job completed in {elapsed:.1f}s")
            print(f"Output PC IDs: {job.get('output_pc_ids', [])}")
            print(f"Output STL IDs: {job.get('output_stl_ids', [])}")
            print(f"Generation time: {job.get('generation_time_ms', 0)/1000:.1f}s")
            return job
        
        if status == "failed":
            print(f"\n✗ Job failed: {job.get('error_message', 'Unknown error')}")
            return None
        
        if status == "cancelled":
            print("\n✗ Job was cancelled")
            return None
        
        time.sleep(3)


def main():
    """Run all tests."""
    print("=" * 60)
    print("  Cloud Generation Integration Test")
    print("=" * 60)
    
    # Test 1: Settings API
    if not test_settings_api():
        print("\n✗ Settings API test failed")
        return 1
    
    # Test 2: System Info
    if not test_system_info():
        print("\n✗ System info test failed")
        return 1
    
    # Test 3: Configure cloud settings
    if not configure_cloud_settings():
        print("\n✗ Failed to configure cloud settings")
        return 1
    
    # Test 4: Get/create project
    project_id = get_or_create_project()
    if not project_id:
        print("\n✗ Failed to get/create project")
        return 1
    
    # Test 5: Get/upload point cloud
    pc_id = get_or_upload_point_cloud(project_id)
    if not pc_id:
        print("\n✗ Failed to get/upload point cloud")
        return 1
    
    # Test 6: Submit cloud generation job
    job_id = submit_cloud_generation_job(project_id, pc_id)
    if not job_id:
        print("\n✗ Failed to submit cloud generation job")
        return 1
    
    # Test 7: Poll for completion
    result = poll_job_status(job_id)
    if not result:
        print("\n✗ Cloud generation job did not complete successfully")
        return 1
    
    print("\n" + "=" * 60)
    print("  All tests passed! ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
