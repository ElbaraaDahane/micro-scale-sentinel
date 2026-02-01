# Micro-Scale Sentinel: Deployment Instructions
## Version 1.0 - January 31, 2026

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Local Development Setup](#2-local-development-setup)
3. [Production Deployment](#3-production-deployment)
4. [Underwater Robot Integration](#4-underwater-robot-integration)
5. [Testing & Validation](#5-testing--validation)
6. [Troubleshooting](#6-troubleshooting)
7. [Maintenance & Updates](#7-maintenance--updates)

---

## 1. SYSTEM REQUIREMENTS

### 1.1 Hardware Requirements

**Minimum Configuration:**
- **CPU**: 2 cores, 2.0 GHz (Intel i5 or equivalent)
- **RAM**: 4 GB
- **Storage**: 10 GB free space (for database and logs)
- **Network**: 5 Mbps internet connection (for Gemini API)
- **Display**: 1280×720 resolution (for dashboard)

**Recommended Configuration:**
- **CPU**: 4+ cores, 3.0 GHz (Intel i7 or equivalent)
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 10+ Mbps internet connection
- **Display**: 1920×1080 resolution
- **GPU**: Optional (for future local model fine-tuning)

**For Underwater Deployment (ROV/AUV Integration):**
- **Tethered ROV**: Surface computer with above specs + ethernet tether
- **Autonomous AUV**: Onboard single-board computer (Raspberry Pi 4+ or NVIDIA Jetson)
  - Min: 4GB RAM, 64GB storage
  - Recommended: 8GB RAM, 128GB storage

### 1.2 Software Requirements

**Operating System:**
- Linux (Ubuntu 20.04+ recommended)
- macOS (10.15+)
- Windows 10/11

**Required Software:**
- **Python**: 3.9 or higher
- **pip**: 21.0 or higher
- **Git**: 2.30 or higher (for cloning repository)
- **SQLite**: 3.35+ (usually included with Python)

**Optional:**
- **Docker**: 24.0+ (for containerized deployment)
- **PostgreSQL**: 13+ (for production database)
- **Redis**: 7.0+ (for caching)
- **Nginx**: 1.20+ (for web dashboard hosting)

---

## 2. LOCAL DEVELOPMENT SETUP

### 2.1 Installation (Step-by-Step)

#### **Step 1: Clone Repository**

```bash
# Clone the repository
git clone https://github.com/your-org/micro-scale-sentinel.git
cd micro-scale-sentinel

# Verify structure
ls -la
```

Expected directory structure:
```
micro-scale-sentinel/
├── config/
│   ├── config.yaml
│   └── .env.example
├── src/
│   ├── preprocessing.py
│   ├── classifier.py
│   ├── storage.py
│   ├── reporting.py
│   └── main.py
├── data/
│   ├── raw_images/
│   ├── preprocessed/
│   ├── results/
│   └── database/
├── tests/
│   ├── test_preprocessing.py
│   ├── test_classifier.py
│   └── ...
├── docs/
│   ├── technical-architecture.md
│   ├── prompt-engineering-guide.md
│   ├── data-format-specs.md
│   └── deployment-instructions.md
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

#### **Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Verify activation (should show venv path)
which python
```

#### **Step 3: Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
pip list
```

**requirements.txt:**
```txt
# Core dependencies
numpy==1.24.3
scipy==1.10.1
pandas==2.0.2
opencv-python==4.8.0.74

# Google Gemini API
google-generativeai==0.3.0

# Visualization
matplotlib==3.7.1
plotly==5.14.1

# Data processing
Pillow==10.0.0

# Database
SQLAlchemy==2.0.19  # ORM for database interactions

# API (optional - for web service)
fastapi==0.100.0
uvicorn==0.23.1

# Configuration
python-dotenv==1.0.0
pyyaml==6.0

# Testing
pytest==7.4.0
pytest-cov==4.1.0

# Utilities
tqdm==4.65.0  # Progress bars
colorlog==6.7.0  # Colored logging
```

#### **Step 4: Configure API Keys**

```bash
# Copy example environment file
cp config/.env.example config/.env

# Edit .env file with your API key
nano config/.env
```

**config/.env:**
```bash
# Google Gemini API Key (REQUIRED)
GEMINI_API_KEY=your_api_key_here

# Database credentials (if using PostgreSQL)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=microplastics
DB_USER=sentinel
DB_PASSWORD=your_password_here

# Optional: Redis for caching
REDIS_HOST=localhost
REDIS_PORT=6379
```

**How to get Gemini API Key:**
1. Visit https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy key and paste into `.env` file

#### **Step 5: Initialize Database**

```bash
# Run database initialization script
python src/init_database.py

# Verify database created
ls -lh data/database/
```

**Expected output:**
```
Creating database at: data/database/sentinel.db
Creating tables...
✓ Table: classifications
✓ Table: particle_features
✓ Table: image_metadata
✓ Table: daily_statistics
✓ Indexes created
Database initialized successfully!
```

#### **Step 6: Test Installation**

```bash
# Run test suite
pytest tests/ -v

# Expected: All tests pass
```

**If tests fail:**
- Check API key is valid
- Verify internet connection
- Ensure all dependencies installed correctly

---

### 2.2 Quick Start: Classify Your First Image

```bash
# Download sample holographic image
wget https://example.com/sample_hologram.png -O data/raw_images/sample.png

# Run classification
python src/main.py --image data/raw_images/sample.png

# View results
cat data/results/classifications/sample_result.json
```

**Expected output:**
```json
{
  "classification": "MICROPLASTIC",
  "confidence_microplastic": 87,
  "polymer_type": "PET",
  "reasoning": "The regular, uniform diffraction pattern..."
}
```

---

## 3. PRODUCTION DEPLOYMENT

### 3.1 Docker Deployment (Recommended)

#### **Step 1: Build Docker Image**

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/raw_images data/preprocessed data/results data/database logs

# Expose port (if running API)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

# Run application
CMD ["python", "src/main.py", "--mode", "server"]
```

**Build and run:**
```bash
# Build image
docker build -t micro-scale-sentinel:latest .

# Run container
docker run -d \
  --name sentinel \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -p 8000:8000 \
  micro-scale-sentinel:latest

# Check logs
docker logs -f sentinel
```

#### **Step 2: Docker Compose (Multi-Service)**

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  sentinel:
    build: .
    container_name: sentinel-app
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - DB_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15
    container_name: sentinel-db
    environment:
      - POSTGRES_DB=microplastics
      - POSTGRES_USER=sentinel
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7
    container_name: sentinel-cache
    ports:
      - "6379:6379"
    restart: unless-stopped

  dashboard:
    image: nginx:latest
    container_name: sentinel-dashboard
    volumes:
      - ./dashboard:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - sentinel
    restart: unless-stopped

volumes:
  postgres_data:
```

**Start services:**
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

### 3.2 Cloud Deployment (Google Cloud Run)

#### **Step 1: Prepare for Cloud Deployment**

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

#### **Step 2: Build and Push Container**

```bash
# Build for Cloud Run
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/micro-scale-sentinel

# Or use local Docker and push
docker build -t gcr.io/YOUR_PROJECT_ID/micro-scale-sentinel .
docker push gcr.io/YOUR_PROJECT_ID/micro-scale-sentinel
```

#### **Step 3: Deploy to Cloud Run**

```bash
# Deploy service
gcloud run deploy micro-scale-sentinel \
  --image gcr.io/YOUR_PROJECT_ID/micro-scale-sentinel \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY

# Get service URL
gcloud run services describe micro-scale-sentinel --region us-central1
```

#### **Step 4: Configure Cloud SQL (PostgreSQL)**

```bash
# Create Cloud SQL instance
gcloud sql instances create sentinel-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database
gcloud sql databases create microplastics --instance=sentinel-db

# Connect Cloud Run to Cloud SQL
gcloud run services update micro-scale-sentinel \
  --add-cloudsql-instances YOUR_PROJECT_ID:us-central1:sentinel-db
```

---

## 4. UNDERWATER ROBOT INTEGRATION

### 4.1 Deployment Options

| Option | Type | Setup Complexity | Cost | Best For |
|--------|------|------------------|------|----------|
| **A** | Tethered ROV | ⭐⭐ Medium | $5-10K | Hackathon/Prototype |
| **B** | Autonomous AUV | ⭐⭐⭐ High | $50-150K | Long-term research |
| **C** | Surface Processing | ⭐ Easy | <$1K | Budget testing |

---

### 4.2 Option A: Tethered ROV (BlueROV2) - RECOMMENDED

**System Architecture:**

```
┌─────────────────────────────────────────────┐
│           UNDERWATER (ROV)                  │
│  ┌──────────────────────────────────────┐  │
│  │  Holographic Microscope               │  │
│  │  • Captures particle images           │  │
│  │  • 2.5 kg payload                     │  │
│  │  • Power: 12V from ROV               │  │
│  └───────────┬──────────────────────────┘  │
│              │ USB/Ethernet                 │
│  ┌───────────▼──────────────────────────┐  │
│  │  Onboard Computer (optional)          │  │
│  │  • Raspberry Pi 4 (8GB)              │  │
│  │  • Stores images temporarily         │  │
│  │  • Forwards to surface via tether    │  │
│  └───────────┬──────────────────────────┘  │
└──────────────┼──────────────────────────────┘
               │ Ethernet Tether (up to 300m)
               │
┌──────────────▼──────────────────────────────┐
│           SURFACE (Boat/Vessel)             │
│  ┌──────────────────────────────────────┐  │
│  │  Surface Computer                     │  │
│  │  • Receives images in real-time      │  │
│  │  • Runs Micro-Scale Sentinel         │  │
│  │  • Connects to internet (Gemini API) │  │
│  │  • Displays dashboard                │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Hardware Requirements:**

**Underwater:**
- BlueROV2 Heavy Configuration (or equivalent)
- Holographic microscope module (2-3 kg)
- Mounting bracket (custom fabrication)
- Power splitter (12V from ROV battery)
- Optional: Raspberry Pi 4 for buffering

**Surface:**
- Laptop/computer with Micro-Scale Sentinel installed
- Tether interface (included with BlueROV2)
- Internet connection (WiFi/4G/satellite)

**Integration Steps:**

#### **Step 1: Mount Holographic Microscope**

```bash
# Physical mounting:
1. Attach microscope to ROV frame using custom bracket
2. Connect power cable to ROV 12V rail
3. Connect data cable (USB or Ethernet) to onboard computer or tether
4. Ensure waterproof connections (IP68 connectors)
5. Test in shallow water (<10m) before deep deployment
```

#### **Step 2: Configure Data Pipeline**

**On ROV (if using Raspberry Pi):**

```bash
# Install lightweight image forwarder
ssh pi@bluerov.local

# Install dependencies
sudo apt-get update
sudo apt-get install python3-opencv python3-numpy

# Create forwarding script
cat > /home/pi/forward_images.py << 'EOF'
import socket
import cv2
import pickle
import struct

# Connect to microscope
cap = cv2.VideoCapture(0)  # USB camera

# Connect to surface computer
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.2.1', 8485))  # Surface computer IP

while True:
    ret, frame = cap.read()
    if ret:
        # Serialize frame
        data = pickle.dumps(frame)
        # Send with length prefix
        client_socket.sendall(struct.pack("Q", len(data)) + data)
EOF

# Start on boot
sudo systemctl enable forward_images.service
```

**On Surface Computer:**

```python
# Receive and process images (src/rov_interface.py)
import socket
import pickle
import struct
from classifier import GeminiClassifier

def receive_images():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8485))
    server_socket.listen(1)
    
    print("Waiting for ROV connection...")
    conn, addr = server_socket.accept()
    print(f"Connected to ROV: {addr}")
    
    classifier = GeminiClassifier(api_key=os.getenv('GEMINI_API_KEY'))
    
    data_buffer = b""
    payload_size = struct.calcsize("Q")
    
    while True:
        # Receive message size
        while len(data_buffer) < payload_size:
            data_buffer += conn.recv(4096)
        
        packed_msg_size = data_buffer[:payload_size]
        data_buffer = data_buffer[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        
        # Receive frame data
        while len(data_buffer) < msg_size:
            data_buffer += conn.recv(4096)
        
        frame_data = data_buffer[:msg_size]
        data_buffer = data_buffer[msg_size:]
        
        # Deserialize frame
        frame = pickle.loads(frame_data)
        
        # Classify
        result = classifier.classify_particle(frame, metadata={})
        
        # Save to database
        storage.save_classification(result)
        
        # Update dashboard
        dashboard.update(result)

if __name__ == "__main__":
    receive_images()
```

**Run on surface computer:**
```bash
python src/rov_interface.py
```

#### **Step 3: Field Deployment Checklist**

**Pre-Dive:**
- [ ] Microscope powered and tested
- [ ] Tether connections secure and waterproof
- [ ] Surface computer running Micro-Scale Sentinel
- [ ] Internet connection active (test Gemini API)
- [ ] Database initialized and accessible
- [ ] Dashboard displaying real-time feed
- [ ] ROV battery charged (4+ hours runtime)
- [ ] Backup power for surface computer

**During Dive:**
- [ ] Monitor image quality on dashboard
- [ ] Check classification results in real-time
- [ ] Verify database is logging results
- [ ] Monitor ROV depth and position (GPS)
- [ ] Take notes on interesting observations

**Post-Dive:**
- [ ] Export data (CSV, JSON)
- [ ] Generate summary report
- [ ] Review flagged uncertain classifications
- [ ] Backup database
- [ ] Rinse ROV and microscope with fresh water

---

### 4.3 Option B: Autonomous AUV

**System Architecture:**

```
┌─────────────────────────────────────────────┐
│        UNDERWATER (AUV - Autonomous)        │
│  ┌──────────────────────────────────────┐  │
│  │  Holographic Microscope               │  │
│  └───────────┬──────────────────────────┘  │
│              │                               │
│  ┌───────────▼──────────────────────────┐  │
│  │  Onboard Computer (NVIDIA Jetson)    │  │
│  │  • Captures images                   │  │
│  │  • Lightweight preprocessing         │  │
│  │  • Stores for post-mission analysis  │  │
│  │  • Optional: Edge classification     │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
             Mission: 8-24 hours
                    │
                    ▼ (After recovery)
┌─────────────────────────────────────────────┐
│        SURFACE (Post-Mission Analysis)      │
│  ┌──────────────────────────────────────┐  │
│  │  Download data from AUV              │  │
│  │  Run full Gemini 3 classification    │  │
│  │  Generate comprehensive reports      │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**AUV Requirements:**
- Hydrus, Seaglider, X-300, or similar
- 2-3 kg payload capacity
- Power budget: 5-10W continuous
- Storage: 128GB+ for image capture
- Onboard computer: NVIDIA Jetson Nano or Xavier

**Workflow:**

1. **Pre-Mission**: Load mission waypoints, start logging
2. **During Mission**: AUV autonomously samples water, captures holograms
3. **Onboard Processing**: Lightweight edge model flags potential microplastics
4. **Post-Mission**: Download data, run full Gemini 3 analysis on shore

**Integration script:**

```python
# Onboard AUV (lightweight edge processing)
import cv2
import numpy as np
from lightweight_model import FastClassifier  # Pre-trained local model

fast_classifier = FastClassifier()

def onboard_processing(frame):
    # Quick pre-screening (60% accuracy, 0.1s)
    quick_result = fast_classifier.predict(frame)
    
    # Save all images + quick classification
    metadata = {
        'timestamp': time.time(),
        'gps': aeuv.get_gps(),
        'depth': auv.get_depth(),
        'quick_classification': quick_result
    }
    
    save_image(frame, metadata)
    
    return quick_result

# Post-mission (full Gemini 3 processing)
def post_mission_analysis(image_dir):
    classifier = GeminiClassifier(api_key=API_KEY)
    
    for image_file in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_file))
        metadata = load_metadata(image_file)
        
        # Full Gemini classification (85%+ accuracy, 3s)
        result = classifier.classify_particle(image, metadata)
        
        storage.save_classification(result)
```

---

### 4.4 Option C: Surface-Based Processing

**Simplest deployment:**

```
┌─────────────────────────────────────────────┐
│          SURFACE (Boat/Pier)                │
│  ┌──────────────────────────────────────┐  │
│  │  Water sampling (bucket/pump)        │  │
│  └───────────┬──────────────────────────┘  │
│              │                               │
│  ┌───────────▼──────────────────────────┐  │
│  │  Portable Holographic Microscope     │  │
│  │  • USB connected to laptop           │  │
│  └───────────┬──────────────────────────┘  │
│              │                               │
│  ┌───────────▼──────────────────────────┐  │
│  │  Laptop running Micro-Scale Sentinel │  │
│  │  • Real-time classification          │  │
│  │  • Instant results                   │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Advantages:**
- Lowest cost (<$1K excluding microscope)
- Immediate results
- Easy to operate
- No waterproofing challenges

**Limitations:**
- Not continuous monitoring
- Requires surface access
- Weather dependent

---

## 5. TESTING & VALIDATION

### 5.1 Unit Tests

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test module
pytest tests/test_classifier.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### 5.2 Integration Testing

```python
# Test end-to-end pipeline (tests/test_integration.py)
def test_end_to_end_classification():
    # Load test image
    image = cv2.imread('tests/fixtures/test_pet_fiber.png')
    metadata = load_test_metadata()
    
    # Preprocess
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.process(image)
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(enhanced, metadata)
    
    # Classify
    classifier = GeminiClassifier(api_key=TEST_API_KEY)
    result = classifier.classify_particle(features[0], enhanced, metadata)
    
    # Verify
    assert result['classification'] == 'MICROPLASTIC'
    assert result['polymer_type'] == 'PET'
    assert result['confidence_microplastic'] > 80
    
    # Store
    storage = StorageManager('tests/test.db')
    storage.save_classification(result, features[0], metadata)
    
    # Verify storage
    retrieved = storage.get_classification(metadata['image_id'], 'P001')
    assert retrieved['classification'] == result['classification']
```

### 5.3 Performance Testing

```bash
# Benchmark classification speed
python tests/benchmark_speed.py --images 100

# Expected output:
# Preprocessing: 0.8s per image
# Feature extraction: 0.3s per image
# Gemini classification: 2.5s per image
# Total: 3.6s per image
# Throughput: 16.7 images/minute
```

---

## 6. TROUBLESHOOTING

### 6.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| API key invalid | `AuthenticationError` | Verify key in `.env`, check https://makersuite.google.com |
| Rate limit exceeded | `429 Too Many Requests` | Wait 1 minute, reduce batch size |
| Image load failure | `cv2.error` | Check image format (PNG/TIFF), verify file exists |
| Low classification accuracy | <70% on test set | Review prompt engineering guide, validate test labels |
| Database locked | `sqlite3.OperationalError` | Close other connections, use PostgreSQL for production |
| Out of memory | System crash | Reduce image resolution, process in batches |

### 6.2 Debugging Commands

```bash
# Check Python environment
python --version
pip list | grep -E "(opencv|numpy|google)"

# Test API connectivity
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print(genai.list_models())"

# Verify database integrity
sqlite3 data/database/sentinel.db "PRAGMA integrity_check;"

# Check disk space
df -h

# Monitor resource usage
htop
```

### 6.3 Logging

```bash
# View real-time logs
tail -f logs/$(date +%Y%m%d).log

# Search for errors
grep "ERROR" logs/*.log

# Filter by component
grep "classifier" logs/*.log | grep "ERROR"
```

---

## 7. MAINTENANCE & UPDATES

### 7.1 Regular Maintenance Tasks

**Daily:**
- [ ] Check dashboard for unusual patterns
- [ ] Review flagged uncertain classifications
- [ ] Monitor API quota usage

**Weekly:**
- [ ] Backup database (`cp data/database/sentinel.db backups/`)
- [ ] Review classification accuracy metrics
- [ ] Update daily statistics table

**Monthly:**
- [ ] Update dependencies (`pip install --upgrade -r requirements.txt`)
- [ ] Review and update prompts based on new failure modes
- [ ] Generate comprehensive reports for stakeholders

**Quarterly:**
- [ ] Re-validate on new ground truth data
- [ ] Consider fine-tuning local models
- [ ] Update documentation

### 7.2 Updating the System

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Run migrations (if database schema changed)
python src/migrate_database.py

# Restart service
docker-compose restart sentinel
# or
systemctl restart sentinel
```

### 7.3 Backup Strategy

**Automated backups:**

```bash
# Create backup script (scripts/backup.sh)
#!/bin/bash
BACKUP_DIR="/backups/sentinel"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup database
cp data/database/sentinel.db "$BACKUP_DIR/sentinel_$TIMESTAMP.db"

# Backup configuration
tar -czf "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" config/

# Backup logs (last 30 days)
tar -czf "$BACKUP_DIR/logs_$TIMESTAMP.tar.gz" logs/

# Remove old backups (keep last 30 days)
find "$BACKUP_DIR" -name "*.db" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $TIMESTAMP"
```

**Schedule with cron:**
```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /path/to/scripts/backup.sh >> /var/log/sentinel_backup.log 2>&1
```

---

## APPENDIX A: Quick Reference Commands

### Deployment Commands

```bash
# Local development
python src/main.py --mode single --image path/to/image.png

# Batch processing
python src/main.py --mode batch --input-dir data/raw_images/ --output-dir data/results/

# Start web server
python src/main.py --mode server --host 0.0.0.0 --port 8000

# Docker
docker-compose up -d
docker-compose logs -f sentinel
docker-compose down

# Cloud (Google Cloud Run)
gcloud run deploy micro-scale-sentinel --image gcr.io/PROJECT/sentinel
```

### Database Commands

```bash
# Initialize database
python src/init_database.py

# Backup database
cp data/database/sentinel.db backups/sentinel_$(date +%Y%m%d).db

# Query database
sqlite3 data/database/sentinel.db "SELECT COUNT(*) FROM classifications;"

# Export to CSV
sqlite3 -header -csv data/database/sentinel.db "SELECT * FROM classifications;" > export.csv
```

---

## APPENDIX B: Deployment Checklist

**Pre-Deployment:**
- [ ] Hardware meets minimum requirements
- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Gemini API key configured in `.env`
- [ ] Database initialized
- [ ] Test classification successful
- [ ] All unit tests pass

**Production Deployment:**
- [ ] Docker container built and tested
- [ ] Database backed up
- [ ] Monitoring configured
- [ ] Backup strategy in place
- [ ] Documentation reviewed
- [ ] Emergency rollback plan prepared

**Underwater Deployment (ROV/AUV):**
- [ ] Holographic microscope tested
- [ ] Waterproof connections verified
- [ ] Power budget validated
- [ ] Data pipeline tested end-to-end
- [ ] Surface computer configured
- [ ] Internet connectivity tested
- [ ] Safety protocols reviewed

---

**Document Version**: 1.0
**Last Updated**: January 31, 2026
**Author**: Micro-Scale Sentinel Team
**Status**: Production Ready