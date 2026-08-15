# Cloud Deployment Guide — AWS EC2 & Docker

## Project: Culturally Aware African & Global Literary Recommender Platform

> **Target Environment:** AWS Elastic Compute Cloud (EC2) / Docker
> **Configuration Files:** `project/Dockerfile`, `project/docker-compose.yml`, `project/requirements.txt`

---

## 1. Cloud Architecture Topology

```
┌─────────────────┐        ┌──────────────────┐        ┌────────────────────────┐
│  Client Browser │───────►│  Nginx / HTTPS   │───────►│ Docker Container       │
│  (React SPA)    │  SSL   │  Reverse Proxy   │ :8000  │                        │
└─────────────────┘  :443  │  (Port 80/443)   │        │ ┌────────────────────┐ │
                           └──────────────────┘        │ │ FastAPI API Server │ │
                                                       │ ├────────────────────┤ │
                                                       │ │ Hybrid Engine      │ │
                                                       │ │ (FM v2 + SVD++)    │ │
                                                       │ ├────────────────────┤ │
                                                       │ │ 50,000 Book Catalog│ │
                                                       │ └────────────────────┘ │
                                                       └────────────────────────┘
```

---

## 2. Hardware Sizing & Recommended Instance Type

| Resource | Minimum Requirement | Recommended AWS Instance |
|:---|:---|:---|
| **vCPU** | 2 vCPUs | **`t3.medium`** (2 vCPU, 4 GiB RAM) |
| **RAM** | 3.5 GiB | **`t3.large`** (2 vCPU, 8 GiB RAM — optimal for multi-worker Uvicorn) |
| **Storage** | 20 GB gp3 SSD | 30 GB gp3 SSD |
| **OS** | Ubuntu 22.04 LTS / Amazon Linux 2023 | Ubuntu 22.04 LTS (x86_64) |

---

## 3. Step-by-Step Deployment Instructions

### Step 1: Launch EC2 Instance on AWS Console
1. Log into your AWS Console and open the **EC2 Dashboard**.
2. Click **Launch Instance**:
   * **Name:** `afriread-recommender-prod`
   * **AMI:** Ubuntu Server 22.04 LTS (HVM), SSD Volume Type.
   * **Instance Type:** `t3.medium` (or `t3.large`).
   * **Key Pair:** Select or create your `.pem` SSH key.
   * **Storage:** 30 GiB General Purpose SSD (gp3).

### Step 2: Configure Inbound Security Group Rules
Ensure the following ports are open in your Security Group:

| Type | Protocol | Port Range | Source | Purpose |
|:---|:---|:---|:---|:---|
| **SSH** | TCP | `22` | Your IP / `0.0.0.0/0` | Administrative Shell Access |
| **HTTP** | TCP | `80` | `0.0.0.0/0` | Web Traffic & Certbot SSL Validation |
| **HTTPS** | TCP | `443` | `0.0.0.0/0` | Secure Web Traffic |
| **Custom TCP** | TCP | `8000` | `0.0.0.0/0` | Direct FastAPI Access (Optional/Dev) |

---

### Step 3: Connect and Install Docker on the EC2 Server
SSH into your instance:
```bash
ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>
```

Update system packages and install Docker:
```bash
# Update packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu
newgrp docker

# Install Docker Compose plugin
sudo apt-get install -y docker-compose-plugin
```

---

### Step 4: Transfer Code or Clone Repository
From your local Mac terminal, upload the project directory:
```bash
# Option A: Upload via rsync / scp
rsync -avz -e "ssh -i your-key.pem" \
  --exclude 'node_modules' --exclude '__pycache__' --exclude 'venv*' \
  /Users/ayomideayanwola/projects/recommenders/movielens/ \
  ubuntu@<your-ec2-public-ip>:~/recommender-app/
```

Or clone via Git on the server:
```bash
git clone <your-repo-url> ~/recommender-app
cd ~/recommender-app
```

---

### Step 5: Build and Run with Docker Compose
On the EC2 instance:
```bash
cd ~/recommender-app/project

# Build and start the container in detached daemon mode
docker compose up --build -d

# Check running container logs
docker compose logs -f
```

Test that the health endpoint responds:
```bash
curl http://localhost:8000/api/health
```

---

### Step 6: Configure Nginx & Free SSL Certificate (Let's Encrypt)
To route standard web traffic (`port 80/443`) to the container with HTTPS:

1. Install Nginx and Certbot:
```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

2. Create an Nginx site configuration (`/etc/nginx/sites-available/recommender`):
```nginx
server {
    server_name your-domain.com www.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

3. Enable the site and obtain SSL:
```bash
sudo ln -s /etc/nginx/sites-available/recommender /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Run certbot for automated HTTPS certificate
sudo certbot --nginx -d your-domain.com
```

---

## 4. Production Operations & Health Monitoring

### Restarting the Application:
```bash
docker compose restart
```

### Checking Application Status:
```bash
docker ps
curl http://localhost:8000/api/health
```

### Viewing Real-Time Logs:
```bash
docker logs -f afriread_recommender_app
```
