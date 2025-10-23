# PCDiff Web Viewer

Interactive 3D web-based viewer for PCDiff skull implant generation inference results. Built with FastAPI (backend) and React + Three.js (frontend).

## Features

- 🔍 **Interactive 3D Visualization**: View defective skulls and generated implants in real-time
- 🎨 **Visual Controls**: Toggle visibility, cycle through ensemble samples, adjust camera
- 📥 **Export Options**: Download PLY (visualization) and STL (3D printing) formats
- 🚀 **Auto-conversion**: Automatically converts NPY files to web formats on-demand
- 🌐 **Remote Access**: Access from external computers via HTTP
- 📊 **Batch Processing**: Convert all inference results at once

## Architecture

```
web_viewer/
├── backend/          # FastAPI server
│   ├── main.py      # REST API endpoints
│   ├── config.py    # Configuration
│   └── requirements.txt
├── frontend/         # React + Vite app
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── hooks/       # React hooks
│   │   ├── services/    # API client
│   │   └── types/       # TypeScript types
│   └── package.json
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm (for frontend development)
- Inference results in `inference_results_ddim50/syn/`

### Option 1: Development Mode (Hot Reload)

Start both backend and frontend with hot-reload:

```bash
cd web_viewer
./start_dev.sh
```

Then open: http://localhost:5173

### Option 2: Production Mode (Backend Only)

For remote access, start only the backend:

```bash
cd web_viewer
./start_server.sh
```

Then access the API at: http://YOUR_IP:8080/api

You'll need to build and serve the frontend separately:

```bash
cd frontend
npm install
npm run build
# Serve the dist/ folder with your web server
```

### Option 3: Manual Start

**Backend:**
```bash
cd web_viewer/backend
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080
```

**Frontend:**
```bash
cd web_viewer/frontend
npm install
npm run dev  # Development
# OR
npm run build  # Production build
```

## Configuration

### Environment Variables

Configure the server using environment variables with the `PCDIFF_` prefix:

```bash
export PCDIFF_HOST="0.0.0.0"                    # Server host
export PCDIFF_PORT=8080                         # Server port
export PCDIFF_EXTERNAL_ORIGIN="http://YOUR_IP:5173"  # Add to CORS
export PCDIFF_AUTO_CONVERT=true                 # Auto-convert on access
export PCDIFF_EXPORT_STL=true                   # Export STL files
```

### Frontend API URL

Set the backend URL in the frontend:

```bash
# Create .env file in frontend/
echo "VITE_API_URL=http://YOUR_IP:8080/api" > frontend/.env
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Health check |
| `/api/results` | GET | List all inference results |
| `/api/results/{id}` | GET | Get specific result metadata |
| `/api/results/{id}/files` | GET | List converted files |
| `/api/convert/{id}` | POST | Convert specific result |
| `/api/convert/batch` | POST | Batch convert all results |
| `/api/files/{id}/{filename}` | GET | Download converted file |
| `/api/conversion-status/{id}` | GET | Check conversion status |

## File Conversion

### Manual Conversion

Convert inference results using the Python script:

```bash
# Convert single result
python3 pcdiff/utils/convert_to_web.py inference_results_ddim50/syn/random_2077_surf

# Batch convert all results
python3 pcdiff/utils/convert_to_web.py inference_results_ddim50 --batch

# Force reconversion
python3 pcdiff/utils/convert_to_web.py inference_results_ddim50 --batch --force

# Skip STL export (faster)
python3 pcdiff/utils/convert_to_web.py inference_results_ddim50 --batch --no-stl
```

### Auto Conversion

The server can automatically convert files when accessed if `auto_convert` is enabled (default: true).

## Remote Access Setup

### 1. Find Your External IP

```bash
# On the server
curl ifconfig.me
# Example output: 34.123.45.67
```

### 2. Configure Firewall

Allow incoming connections on port 8080:

```bash
# GCP
gcloud compute firewall-rules create pcdiff-web --allow tcp:8080

# AWS
aws ec2 authorize-security-group-ingress --group-id sg-xxx --protocol tcp --port 8080 --cidr 0.0.0.0/0

# Ubuntu/Debian
sudo ufw allow 8080/tcp
```

### 3. Start Server

```bash
cd web_viewer
PCDIFF_HOST="0.0.0.0" PCDIFF_PORT=8080 ./start_server.sh
```

### 4. Access from External Computer

Open in your browser:
```
http://34.123.45.67:8080/api/status
```

## Usage

### Viewing Results

1. Select a result from the left panel
2. The 3D viewer will load the defective skull (gray) and generated implant (red)
3. Use mouse controls:
   - **Left click + drag**: Rotate camera
   - **Right click + drag**: Pan camera
   - **Scroll wheel**: Zoom in/out

### Controls

- **Toggle Input/Sample**: Show or hide defective skull and implant
- **Sample Selection**: Cycle through ensemble samples (if multiple exist)
- **Reset Camera**: Return to default view

### Downloads

- **PLY files**: For visualization in 3D software (MeshLab, CloudCompare, etc.)
- **STL files**: For 3D printing slicers (PrusaSlicer, Cura, etc.)

## Troubleshooting

### "Module not found" errors

Install missing Python dependencies:
```bash
pip install fastapi uvicorn pydantic pydantic-settings trimesh
```

### CORS errors in browser

Add your frontend URL to CORS origins:
```bash
export PCDIFF_EXTERNAL_ORIGIN="http://YOUR_IP:5173"
```

### Files not loading

1. Check if files are converted:
   ```bash
   ls inference_results_ddim50/syn/*/web/
   ```

2. Trigger manual conversion:
   ```bash
   python3 pcdiff/utils/convert_to_web.py inference_results_ddim50 --batch
   ```

### Port already in use

Change the port:
```bash
PCDIFF_PORT=8081 ./start_server.sh
```

## Development

### Adding New Features

1. **Backend**: Edit `backend/main.py` to add new API endpoints
2. **Frontend**: Add components in `frontend/src/components/`
3. **API Client**: Update `frontend/src/services/api.ts`

### Building for Production

```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/` and can be served with any web server (nginx, Apache, etc.).

## License

Same as PCDiff project.

## Credits

- **PCDiff**: Point Cloud Diffusion Models for Skull Implant Generation
- **Three.js**: 3D visualization
- **React Three Fiber**: React bindings for Three.js
- **FastAPI**: Modern Python web framework

