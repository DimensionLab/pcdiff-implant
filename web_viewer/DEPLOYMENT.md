# PCDiff Web Viewer - Deployment Guide

This guide covers deploying the PCDiff Web Viewer for remote access from external computers.

## Table of Contents

1. [Server Setup](#server-setup)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Starting the Server](#starting-the-server)
5. [Firewall Configuration](#firewall-configuration)
6. [Frontend Deployment](#frontend-deployment)
7. [HTTPS/SSL Setup](#httpsssl-setup)
8. [Troubleshooting](#troubleshooting)

---

## Server Setup

### System Requirements

- Ubuntu 20.04+ or similar Linux distribution
- Python 3.10+
- 4GB+ RAM
- 20GB+ free disk space (for inference results)
- Public IP address or domain name

### Find Your Server IP

```bash
# Internal IP
hostname -I

# External/Public IP
curl ifconfig.me
# Example: 34.123.45.67
```

---

## Installation

### 1. Install Python Dependencies

```bash
cd /home/michaltakac/pcdiff-implant

# Install backend dependencies
pip install fastapi uvicorn pydantic pydantic-settings aiofiles

# Install conversion dependencies
pip install trimesh numpy
```

### 2. Install Node.js (for frontend development/building)

```bash
# Using NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

### 3. Build Frontend

```bash
cd web_viewer/frontend

# Install dependencies
npm install

# Build for production
npm run build

# Output will be in dist/
```

---

## Configuration

### Backend Configuration

Create environment configuration:

```bash
# Create .env file (optional)
cat > web_viewer/backend/.env << EOF
PCDIFF_HOST=0.0.0.0
PCDIFF_PORT=8080
PCDIFF_AUTO_CONVERT=true
PCDIFF_EXPORT_STL=true
EOF
```

### Frontend Configuration

Update API endpoint:

```bash
# Create frontend .env file
cat > web_viewer/frontend/.env << EOF
VITE_API_URL=http://YOUR_EXTERNAL_IP:8080/api
EOF

# Rebuild after changing .env
cd web_viewer/frontend
npm run build
```

---

## Starting the Server

### Option 1: Production Server (Recommended)

```bash
cd web_viewer
./start_server.sh
```

This will:
- Check dependencies
- Start FastAPI on 0.0.0.0:8080
- Enable external access

### Option 2: Manual Start with Custom Settings

```bash
cd web_viewer/backend
PCDIFF_HOST="0.0.0.0" PCDIFF_PORT=8080 python3 -m uvicorn main:app --workers 4
```

### Option 3: Using systemd (Auto-start on boot)

Create systemd service:

```bash
sudo tee /etc/systemd/system/pcdiff-viewer.service << EOF
[Unit]
Description=PCDiff Web Viewer
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/michaltakac/pcdiff-implant/web_viewer/backend
Environment="PCDIFF_HOST=0.0.0.0"
Environment="PCDIFF_PORT=8080"
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable pcdiff-viewer
sudo systemctl start pcdiff-viewer

# Check status
sudo systemctl status pcdiff-viewer

# View logs
sudo journalctl -u pcdiff-viewer -f
```

---

## Firewall Configuration

### Google Cloud Platform (GCP)

```bash
# Create firewall rule
gcloud compute firewall-rules create pcdiff-web-viewer \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --description "PCDiff Web Viewer"

# Verify
gcloud compute firewall-rules list | grep pcdiff
```

### Amazon Web Services (AWS)

```bash
# Add inbound rule to security group
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 8080 \
  --cidr 0.0.0.0/0
```

### Ubuntu/Debian (UFW)

```bash
# Allow port 8080
sudo ufw allow 8080/tcp

# Check status
sudo ufw status
```

### CentOS/RHEL (firewalld)

```bash
# Add port
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# Verify
sudo firewall-cmd --list-ports
```

---

## Frontend Deployment

### Option 1: Serve with FastAPI (Simple)

Serve the built frontend directly from FastAPI:

```python
# Add to backend/main.py after creating the app
from fastapi.staticfiles import StaticFiles

# Serve frontend static files
app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="frontend")
```

Then access at: `http://YOUR_IP:8080/`

### Option 2: Nginx (Production)

Install and configure Nginx:

```bash
# Install nginx
sudo apt install nginx

# Create nginx config
sudo tee /etc/nginx/sites-available/pcdiff-viewer << EOF
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    # Frontend
    location / {
        root /home/michaltakac/pcdiff-implant/web_viewer/frontend/dist;
        try_files \$uri \$uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/pcdiff-viewer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

Now access at: `http://YOUR_IP/`

### Option 3: Apache (Alternative)

```bash
# Install apache
sudo apt install apache2

# Enable required modules
sudo a2enmod proxy proxy_http headers rewrite

# Create config
sudo tee /etc/apache2/sites-available/pcdiff-viewer.conf << EOF
<VirtualHost *:80>
    ServerName YOUR_DOMAIN_OR_IP

    DocumentRoot /home/michaltakac/pcdiff-implant/web_viewer/frontend/dist

    <Directory /home/michaltakac/pcdiff-implant/web_viewer/frontend/dist>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
        RewriteEngine On
        RewriteCond %{REQUEST_FILENAME} !-f
        RewriteCond %{REQUEST_FILENAME} !-d
        RewriteRule . /index.html [L]
    </Directory>

    ProxyPreserveHost On
    ProxyPass /api http://localhost:8080/api
    ProxyPassReverse /api http://localhost:8080/api
</VirtualHost>
EOF

# Enable site
sudo a2ensite pcdiff-viewer
sudo systemctl reload apache2
```

---

## HTTPS/SSL Setup

### Using Let's Encrypt (Free SSL)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate (for nginx)
sudo certbot --nginx -d YOUR_DOMAIN

# Auto-renewal is configured automatically
# Test renewal
sudo certbot renew --dry-run
```

### Using Self-Signed Certificate (Development)

```bash
# Generate certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/pcdiff.key \
  -out /etc/ssl/certs/pcdiff.crt

# Update nginx config
sudo nano /etc/nginx/sites-available/pcdiff-viewer
# Add:
# listen 443 ssl;
# ssl_certificate /etc/ssl/certs/pcdiff.crt;
# ssl_certificate_key /etc/ssl/private/pcdiff.key;

sudo systemctl reload nginx
```

---

## Troubleshooting

### Server Not Accessible from External IP

1. **Check if server is running:**
   ```bash
   curl http://localhost:8080/api/status
   ```

2. **Check if port is open:**
   ```bash
   sudo netstat -tulpn | grep 8080
   ```

3. **Test from server itself:**
   ```bash
   curl http://$(curl -s ifconfig.me):8080/api/status
   ```

4. **Check firewall rules:**
   ```bash
   sudo ufw status
   sudo iptables -L -n | grep 8080
   ```

### CORS Errors

Add your external IP to CORS origins:

```bash
export PCDIFF_EXTERNAL_ORIGIN="http://YOUR_EXTERNAL_IP:5173"
```

Or edit `backend/config.py` and add to `cors_origins` list.

### Files Not Loading

1. **Check if files exist:**
   ```bash
   ls -lh inference_results_ddim50/syn/*/web/
   ```

2. **Manually convert:**
   ```bash
   python3 pcdiff/utils/convert_to_web.py inference_results_ddim50 --batch
   ```

3. **Check permissions:**
   ```bash
   chmod -R 755 inference_results_ddim50/
   ```

### High Memory Usage

Reduce number of workers:

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 2
```

### Port Already in Use

Find and kill process:

```bash
sudo lsof -i :8080
sudo kill -9 PID
```

Or use different port:

```bash
PCDIFF_PORT=8081 ./start_server.sh
```

---

## Performance Optimization

### 1. Pre-convert All Results

Convert all results before starting server:

```bash
python3 pcdiff/utils/convert_to_web.py inference_results_ddim50 --batch
```

### 2. Enable Gzip Compression (nginx)

Add to nginx config:

```nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript application/ply application/sla;
gzip_min_length 1000;
```

### 3. Cache Static Files (nginx)

```nginx
location ~* \.(ply|stl)$ {
    expires 1d;
    add_header Cache-Control "public, immutable";
}
```

---

## Security Best Practices

### 1. Restrict Access by IP

```bash
# UFW
sudo ufw allow from YOUR_CLIENT_IP to any port 8080

# Nginx
# Add to server block:
allow YOUR_CLIENT_IP;
deny all;
```

### 2. Add Basic Authentication

```bash
# Install apache2-utils
sudo apt install apache2-utils

# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd username

# Add to nginx location block:
auth_basic "Restricted Access";
auth_basic_user_file /etc/nginx/.htpasswd;
```

### 3. Rate Limiting (nginx)

```nginx
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        location /api {
            limit_req zone=api burst=20;
            proxy_pass http://localhost:8080;
        }
    }
}
```

---

## Monitoring

### Check Server Health

```bash
# API status
curl http://localhost:8080/api/status

# System resources
htop

# Disk space
df -h

# Network connections
sudo netstat -tuln | grep 8080
```

### View Logs

```bash
# Systemd service logs
sudo journalctl -u pcdiff-viewer -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## Backup and Maintenance

### Backup Converted Files

```bash
# Backup all web directories
tar -czf pcdiff-web-backup-$(date +%Y%m%d).tar.gz \
  inference_results_ddim50/syn/*/web/

# Backup to remote server
rsync -avz inference_results_ddim50/syn/*/web/ \
  user@backup-server:/backups/pcdiff-web/
```

### Clean Up Old Files

```bash
# Remove all converted files (will be regenerated on demand)
find inference_results_ddim50/syn/*/web/ -type f -delete

# Remove specific file types
find inference_results_ddim50/syn/*/web/ -name "*.stl" -delete
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review server logs
3. Check GitHub issues
4. Contact project maintainers

---

## Quick Reference

### Start Server
```bash
cd web_viewer && ./start_server.sh
```

### Stop Server
```bash
# If using systemd
sudo systemctl stop pcdiff-viewer

# If running manually
pkill -f "uvicorn main:app"
```

### Convert Files
```bash
python3 pcdiff/utils/convert_to_web.py inference_results_ddim50 --batch
```

### Check Status
```bash
curl http://localhost:8080/api/status
```

### View Logs
```bash
sudo journalctl -u pcdiff-viewer -f
```

