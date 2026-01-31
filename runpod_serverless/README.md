# Runpod Serverless Deployment for PCDiff + Voxelization

This directory contains everything needed to deploy the skull implant generation pipeline as a Runpod Serverless endpoint.

## Overview

The serverless endpoint provides a complete inference pipeline:
1. **PCDiff**: Generates implant point cloud from defective skull
2. **Voxelization**: Converts point cloud to volumetric mesh
3. **S3 Upload**: Uploads results to AWS S3 for retrieval

## Files

- `rp_handler.py` - Main Runpod serverless handler
- `Dockerfile` - Docker image configuration
- `requirements.txt` - Python dependencies
- `build_and_push.sh` - Helper script to build and push Docker image
- `test_client.py` - Python client for testing the deployed endpoint
- `README.md` - This documentation

## Model Loading Priority

The handler supports multiple model sources with the following priority:

1. **Network Volume** (`/runpod-volume/models/`) - **Recommended for active development**
   - Best for frequent model updates without rebuilding Docker
   - Simply upload new models to the network volume
   - Workers will use updated models on next cold start

2. **Environment Variable Override**
   - Set `PCDIFF_MODEL_PATH` or `VOXELIZATION_MODEL_PATH`
   - Useful for testing specific model versions

3. **Embedded in Docker Image** (`/app/models/`)
   - Default fallback
   - Models baked into the Docker image at build time

## Prerequisites

1. **Docker** installed locally
2. **Docker Hub** account (or other container registry)
3. **Runpod** account with API key
4. **AWS S3** bucket for storing results
5. **Model checkpoints**:
   - PCDiff model: `pcdiff/checkpoints/model_best.pth`
   - Voxelization model: `voxelization/checkpoints/model_best.pt`

## Deployment Steps

### 1. Build Docker Image

```bash
cd /path/to/pcdiff-implant

# Build for Runpod (linux/amd64 platform)
docker build --platform linux/amd64 \
  -t YOUR_DOCKERHUB_USERNAME/pcdiff-implant-serverless:v1.0 \
  -f runpod_serverless/Dockerfile .
```

### 2. Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push the image
docker push YOUR_DOCKERHUB_USERNAME/pcdiff-implant-serverless:v1.0
```

### 3. Create Runpod Serverless Endpoint

1. Go to [Runpod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Click **Import from Docker Registry**
4. Enter your image URL: `docker.io/YOUR_USERNAME/pcdiff-implant-serverless:v1.0`
5. Configure settings:
   - **Endpoint Name**: `pcdiff-implant-inference`
   - **GPU Type**: Select 16GB+ GPU (RTX 4090, A10G, L4, etc.)
   - **Container Disk**: 20 GB (for model storage)
   - **Max Workers**: 3 (adjust based on load)
   - **Idle Timeout**: 5 seconds

6. Add **Environment Variables**:
   ```
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_S3_BUCKET=your-bucket-name
   AWS_S3_REGION=us-east-1
   ```

7. Click **Deploy Endpoint**

## Updating Models (Without Rebuilding Docker)

### Option 1: Network Volume (Recommended)

This is the best approach when your model is still training and you need frequent updates.

#### Setup Network Volume

1. Go to [Runpod Storage](https://www.runpod.io/console/storage)
2. Click **Create Network Volume**
3. Choose a datacenter (must match your endpoint's region)
4. Set size (at least 1 GB for models)
5. Name it (e.g., `pcdiff-models`)

#### Upload Models to Network Volume

**Option A: Using Runpod S3-Compatible API**

```bash
# Install AWS CLI if not already installed
pip install awscli

# Configure with Runpod credentials (get from Runpod console)
aws configure --profile runpod
# Enter your Runpod S3 credentials

# Upload models
aws s3 cp pcdiff/checkpoints/model_best.pth \
    s3://YOUR_VOLUME_ID/models/pcdiff_best.pth \
    --endpoint-url https://YOUR_DATACENTER.runpod.io \
    --profile runpod

aws s3 cp voxelization/checkpoints/model_best.pt \
    s3://YOUR_VOLUME_ID/models/voxelization_best.pt \
    --endpoint-url https://YOUR_DATACENTER.runpod.io \
    --profile runpod
```

**Option B: Using a Pod**

1. Create a temporary Pod with the network volume attached
2. Upload models via SSH/SFTP
3. Models will be at `/workspace/models/`
4. Terminate the Pod

#### Attach Network Volume to Endpoint

1. Go to your endpoint settings
2. Click **Edit Endpoint**
3. Expand **Advanced** section
4. Select your network volume under **Network Volume**
5. Save changes

Workers will now load models from `/runpod-volume/models/` on startup.

### Option 2: Rebuild Docker Image

When you have a stable model version:

```bash
# Rebuild with new models
docker build --platform linux/amd64 \
  -t YOUR_USERNAME/pcdiff-implant-serverless:v2.0 \
  -f runpod_serverless/Dockerfile .

# Push new version
docker push YOUR_USERNAME/pcdiff-implant-serverless:v2.0

# Update endpoint in Runpod console to use new image tag
```

### Option 3: GitHub Integration (Auto-Deploy)

For automatic deployments when you push to GitHub:

1. Connect your GitHub repo to Runpod
2. Configure the endpoint to deploy from GitHub
3. Push changes to trigger automatic rebuilds

See [Runpod GitHub Integration](https://docs.runpod.io/serverless/workers/github-integration) for details.

## API Usage

### Request Format

```json
{
  "input": {
    "defective_skull": "<base64_encoded_npy_file>",
    "input_format": "base64",
    "num_ensemble": 1,
    "sampling_steps": 1000,
    "output_prefix": "job_123"
  }
}
```

Or with S3 URL:

```json
{
  "input": {
    "defective_skull": "s3://your-bucket/input/defective_skull.npy",
    "input_format": "s3_url",
    "num_ensemble": 1,
    "sampling_steps": 1000,
    "output_prefix": "job_123"
  }
}
```

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `defective_skull` | string | Yes | - | Base64-encoded .npy file or S3 URL |
| `input_format` | string | No | `base64` | `base64` or `s3_url` |
| `num_ensemble` | int | No | `1` | Number of ensemble samples |
| `sampling_steps` | int | No | `1000` | Diffusion sampling steps |
| `output_prefix` | string | No | random UUID | Prefix for S3 output keys |

### Response Format

```json
{
  "status": "success",
  "results": {
    "skull_complete_ply": "https://bucket.s3.region.amazonaws.com/...",
    "skull_complete_stl": "https://...",
    "implant_only_ply": "https://...",
    "implant_only_stl": "https://...",
    "implant_pc_ply": "https://...",
    "skull_complete_nrrd": "https://...",
    "implant_volume_nrrd": "https://...",
    "implant_npy": "https://..."
  },
  "metadata": {
    "processing_time_seconds": 123.45,
    "num_implant_points": 3072,
    "num_ensemble": 1,
    "sampling_steps": 1000,
    "mesh_vertices": 50000,
    "mesh_faces": 100000,
    "model_source": "network_volume",
    "pcdiff_model": "/runpod-volume/models/pcdiff_best.pth",
    "voxelization_model": "/runpod-volume/models/voxelization_best.pt"
  }
}
```

### Example: Python Client

```python
import runpod
import base64
import numpy as np
import io

# Initialize Runpod client
runpod.api_key = "YOUR_RUNPOD_API_KEY"

# Load and encode input
defective_skull = np.load("path/to/defective_skull.npy")

# Save to bytes buffer for proper .npy format
buffer = io.BytesIO()
np.save(buffer, defective_skull)
encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Submit job
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")
run_request = endpoint.run({
    "input": {
        "defective_skull": encoded,
        "input_format": "base64",
        "num_ensemble": 1,
        "sampling_steps": 1000
    }
})

# Wait for result (or use async)
result = run_request.output()
print(result)
```

### Example: cURL

```bash
# Encode the numpy file
ENCODED=$(base64 -i defective_skull.npy)

# Submit async job
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"defective_skull\": \"$ENCODED\",
      \"input_format\": \"base64\",
      \"num_ensemble\": 1,
      \"sampling_steps\": 1000
    }
  }"

# Check job status
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

## Output Files

| File | Description |
|------|-------------|
| `skull_complete.ply` | Complete skull mesh (defective + implant) in PLY format |
| `skull_complete.stl` | Complete skull mesh in STL format (3D printable) |
| `implant_only.ply` | Implant-only mesh in PLY format |
| `implant_only.stl` | Implant-only mesh in STL format (3D printable) |
| `implant_pc.ply` | Implant point cloud in PLY format |
| `skull_complete.nrrd` | Complete skull as 512³ voxel volume |
| `implant_volume.nrrd` | Implant-only as 512³ voxel volume |
| `implant.npy` | Implant point cloud as numpy array |

## GPU Requirements

- **Minimum**: 16 GB VRAM (RTX 4090, A10G, L4)
- **Recommended**: 24 GB VRAM (A10, RTX 3090)
- **Processing Time**: ~2-5 minutes per inference

## Cost Estimation

- **GPU Cost**: ~$0.30-0.50 per inference (depending on GPU type)
- **S3 Storage**: ~$0.023/GB/month
- **Data Transfer**: ~$0.09/GB (S3 to internet)
- **Network Volume**: ~$0.07/GB/month

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `num_ensemble` or use larger GPU
2. **Slow Cold Start**: Models are loaded on first request (~30-60s)
3. **S3 Upload Fails**: Check AWS credentials and bucket permissions
4. **Model Not Found**: Check network volume is attached and models are uploaded

### Checking Model Source

The response metadata includes `model_source` which indicates where models were loaded from:
- `network_volume`: Models loaded from `/runpod-volume/models/`
- `embedded`: Models loaded from Docker image `/app/models/`

### Logs

View worker logs in Runpod console under your endpoint's **Logs** tab.

## Local Testing

```bash
# Build image locally
docker build -t pcdiff-serverless-test -f runpod_serverless/Dockerfile .

# Run locally (requires NVIDIA Docker)
docker run --gpus all -it \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_S3_BUCKET=your_bucket \
  pcdiff-serverless-test
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Runpod Serverless                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Request   │───▶│  rp_handler │───▶│   Response  │     │
│  │   (JSON)    │    │    .py      │    │   (JSON)    │     │
│  └─────────────┘    └──────┬──────┘    └─────────────┘     │
│                            │                                │
│         ┌──────────────────┼──────────────────┐            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   PCDiff    │    │ Voxelization│    │  S3 Upload  │     │
│  │   Model     │    │    Model    │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Model Sources (Priority Order):                    │   │
│  │  1. Network Volume: /runpod-volume/models/          │   │
│  │  2. Environment Variable Override                   │   │
│  │  3. Embedded: /app/models/                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                               │
│                            ▼                               │
│                     ┌─────────────┐                        │
│                     │  AWS S3     │                        │
│                     │  Bucket     │                        │
│                     └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Updating Your Web App

To use the serverless endpoint from your web app instead of local CPU inference:

1. Get your endpoint ID and API key from Runpod
2. Update your web app backend to call the Runpod API
3. Handle async job submission and polling for results

Example integration:

```python
import runpod
import time

def generate_implant_serverless(defective_skull_npy, api_key, endpoint_id):
    """Call Runpod serverless endpoint for implant generation."""
    runpod.api_key = api_key
    endpoint = runpod.Endpoint(endpoint_id)
    
    # Encode input
    import base64
    import io
    buffer = io.BytesIO()
    np.save(buffer, defective_skull_npy)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Submit job
    run = endpoint.run({
        "input": {
            "defective_skull": encoded,
            "input_format": "base64",
            "num_ensemble": 1,
            "sampling_steps": 1000
        }
    })
    
    # Wait for completion (with timeout)
    result = run.output(timeout=600)
    
    if result.get("status") == "success":
        return result["results"]
    else:
        raise Exception(result.get("error", "Unknown error"))
```

## License

MIT License - See project root LICENSE file.
