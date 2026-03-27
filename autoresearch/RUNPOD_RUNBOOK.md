# RunPod Autoresearch Runbook

Operational knowledge for deploying and running autoresearch experiments on RunPod.

## Pod Creation Rules

### Always Use Secure Cloud
- **Cloud type:** `SECURE` only. Never use `COMMUNITY`.
- **Why:** Community cloud pods get private IPs (100.65.x.x) with no SSH over TCP. Secure cloud provides public IPs with SSH on port 22.
- Cost is higher ($0.46/hr for RTX 3090 secure vs $0.22/hr community) but community is unusable for SSH access.

### Network Volume (CA-MTL-3) is Unreliable
- Network volume `wzcc20z2mg` ("cranial-implant", 60GB) is in CA-MTL-3 datacenter.
- CA-MTL-3 has had **zero GPU availability** across all types (RTX 3090, 4090, A40, L40S, A6000, etc.) for extended periods.
- **Workaround:** Create pods WITHOUT the network volume in any datacenter, then upload the dataset via SCP.
- The SkullBreak dataset is ~6.7GB compressed. SCP upload takes ~2 minutes to most datacenters.

### Working Pod Configuration

```python
# GraphQL mutation that works
mutation = """
mutation {
  podFindAndDeployOnDemand(input: {
    name: "pcdiff-autoresearch"
    gpuTypeId: "NVIDIA GeForce RTX 3090"
    gpuCount: 1
    volumeInGb: 50
    containerDiskInGb: 30
    cloudType: SECURE
    imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    volumeMountPath: "/workspace"
    startSsh: true
    ports: "22/tcp,8888/http"
    env: [{key: "PUBLIC_KEY", value: "<ssh-pub-key>"}]
  }) {
    id name desiredStatus
    machine { gpuDisplayName location }
  }
}
"""
```

Key parameters:
- `cloudType: SECURE` -- required for SSH
- `ports: "22/tcp,8888/http"` -- must explicitly expose SSH TCP port
- `imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"` -- RunPod native image, starts faster than upstream pytorch images
- `PUBLIC_KEY` env var -- inject SSH public key from `/home/mike/.ssh/id_ed25519.pub`
- Do NOT set `networkVolumeId` or `dataCenterId` unless CA-MTL-3 has confirmed availability

### What Does NOT Work

| Configuration | Failure Mode |
|---|---|
| `cloudType: COMMUNITY` | Private IP only, no SSH TCP access |
| `networkVolumeId: wzcc20z2mg` | SUPPLY_CONSTRAINT -- no GPUs in CA-MTL-3 |
| `imageName: "pytorch/pytorch:2.5.x"` | Container stuck provisioning, never reaches RUNNING |
| Missing `ports: "22/tcp"` | SSH port not exposed, only HTTP ports visible |
| `dataCenterId: "CA-MTL-3"` | No GPU availability (as of March 2026) |

### GPU Preferences (Secure Cloud Pricing)

| GPU | Secure $/hr | VRAM | Notes |
|---|---|---|---|
| RTX 3090 | $0.46 | 24GB | Best value for autoresearch |
| RTX 4090 | $0.59 | 24GB | Faster, good alternative |
| A40 | $0.40 | 48GB | Large VRAM if needed |
| L40S | $0.86 | 48GB | Expensive but capable |

## Pod Setup After Creation

### 1. Wait for SSH
Poll the pod API until `runtime.ports` shows a port with `privatePort: 22, type: "tcp", isIpPublic: true`. Typical startup: 60-90 seconds.

```bash
# Check pod status
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://api.runpod.io/graphql" \
  -d '{"query":"query { pod(input: {podId: \"<POD_ID>\"}) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }"}'
```

### 2. SSH Connection
```bash
ssh root@<PUBLIC_IP> -p <PUBLIC_PORT> -i ~/.ssh/id_ed25519
```

### 3. Bootstrap Environment
```bash
# On the pod:
cd /workspace
git clone https://github.com/DimensionLab/pcdiff-implant.git
cd pcdiff-implant
pip install -r requirements.txt
pip install open3d
```

### 4. Upload Dataset (if no network volume)
From the dev server:
```bash
cd /home/mike/pcdiff-implant/datasets
tar czf /tmp/skullbreak_full.tar.gz SkullBreak/
scp -i ~/.ssh/id_ed25519 -P <SSH_PORT> /tmp/skullbreak_full.tar.gz root@<IP>:/workspace/
ssh root@<IP> -p <SSH_PORT> 'cd /workspace/pcdiff-implant/pcdiff/datasets && tar xzf /workspace/skullbreak_full.tar.gz && rm /workspace/skullbreak_full.tar.gz'
```

Or use the upload script:
```bash
python3 autoresearch/upload_data.py --host <IP> --port <SSH_PORT>
```

### 5. Verify Setup
```bash
# On the pod:
cd /workspace/pcdiff-implant
python autoresearch/prepare_pcdiff.py  # Should show "10 eval cases"
```

### 6. Run Experiments
```bash
cd /workspace/pcdiff-implant/autoresearch
OPENROUTER_API_KEY=<key> python run_experiments.py --time-budget 900 --max-experiments 50
```

## API Authentication
- API key: stored in `/home/mike/pcdiff-implant/web_viewer/.env` as `RUNPOD_API_KEY`
- Auth header: `Authorization: Bearer <key>` (NOT `api-key:` header)
- GraphQL endpoint: `https://api.runpod.io/graphql`
- Balance check: query `myself { clientBalance }`

## Monitoring
- Check balance: `myself { clientBalance }` -- budget ~$147 as of 2026-03-27
- List pods: `myself { pods { id name desiredStatus runtime { uptimeInSeconds } } }`
- Stop pod: `mutation { podStop(input: {podId: "<id>"}) { id } }`
- Terminate pod: `mutation { podTerminate(input: {podId: "<id>"}) }`
