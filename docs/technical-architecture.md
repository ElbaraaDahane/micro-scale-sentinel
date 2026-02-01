# Micro-Scale Sentinel: Technical Architecture Document
## Version 1.0 - January 31, 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Component Specifications](#3-component-specifications)
4. [Data Pipeline](#4-data-pipeline)
5. [API Integration](#5-api-integration)
6. [Security & Privacy](#6-security--privacy)
7. [Performance Requirements](#7-performance-requirements)
8. [Scalability Design](#8-scalability-design)

---

## 1. SYSTEM OVERVIEW

### 1.1 Architecture Philosophy

Micro-Scale Sentinel follows a **modular, loosely-coupled architecture** with clear separation of concerns:

- **Preprocessing Module**: Image enhancement and feature extraction (independent)
- **Classification Module**: Gemini 3 API integration (cloud-based)
- **Storage Module**: Results database and caching (local/cloud)
- **Reporting Module**: Analytics and visualization (independent)

**Design Principles:**
- Modularity: Each component can be developed, tested, and replaced independently
- Scalability: Horizontal scaling via parallel processing
- Reliability: Graceful degradation and fallback mechanisms
- Observability: Comprehensive logging and monitoring

### 1.2 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.9+ | Primary development language |
| **AI/ML** | Google Gemini 3 Pro API | Latest | Multimodal classification |
| **Image Processing** | OpenCV | 4.8+ | Image preprocessing and feature extraction |
| **Scientific Computing** | NumPy | 1.24+ | Numerical operations |
| **Scientific Computing** | SciPy | 1.10+ | Scientific algorithms |
| **Data Processing** | Pandas | 2.0+ | Tabular data manipulation |
| **Visualization** | Matplotlib | 3.7+ | Static plotting |
| **Visualization** | Plotly | 5.14+ | Interactive dashboards |
| **Storage** | SQLite / PostgreSQL | - | Structured data storage |
| **Caching** | Redis (optional) | 7.0+ | Result caching for performance |
| **API Framework** | FastAPI (optional) | 0.100+ | REST API for web integration |
| **Deployment** | Docker | 24+ | Containerization |
| **Cloud** | Google Cloud Run | - | Serverless deployment |

---

## 2. ARCHITECTURE DESIGN

### 2.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Holographic  │  │   Metadata   │  │ Environmental│         │
│  │   Images     │  │   (JSON/CSV) │  │     Data     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Image Enhancement Module                              │    │
│  │  • Contrast enhancement (CLAHE)                        │    │
│  │  • Noise reduction (bilateral filter)                  │    │
│  │  • Image normalization                                 │    │
│  └────────────────┬───────────────────────────────────────┘    │
│                   │                                             │
│  ┌────────────────▼───────────────────────────────────────┐    │
│  │  Feature Extraction Module                             │    │
│  │  • Particle detection (contour finding)                │    │
│  │  • Size measurement (area, perimeter)                  │    │
│  │  • Shape analysis (circularity, aspect ratio)          │    │
│  │  • Refractive index estimation (fringe spacing)        │    │
│  │  • Texture features (intensity variance, entropy)      │    │
│  └────────────────┬───────────────────────────────────────┘    │
└───────────────────┼─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CLASSIFICATION LAYER                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Gemini 3 Integration Module                           │    │
│  │  • Multimodal prompt construction                      │    │
│  │  • API request management                              │    │
│  │  • Response parsing and validation                     │    │
│  │  • Error handling and retry logic                      │    │
│  └────────────────┬───────────────────────────────────────┘    │
│                   │                                             │
│  ┌────────────────▼───────────────────────────────────────┐    │
│  │  Result Interpretation Module                          │    │
│  │  • Confidence scoring                                  │    │
│  │  • Hypothesis ranking                                  │    │
│  │  • Uncertainty flagging                                │    │
│  └────────────────┬───────────────────────────────────────┘    │
└───────────────────┼─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Database Module                                        │    │
│  │  • Classification results (structured)                 │    │
│  │  • Feature vectors (for future ML)                     │    │
│  │  • Metadata and timestamps                             │    │
│  └────────────────┬───────────────────────────────────────┘    │
│                   │                                             │
│  ┌────────────────▼───────────────────────────────────────┐    │
│  │  Caching Module (Optional)                             │    │
│  │  • Recent classification cache                         │    │
│  │  • Feature extraction cache                            │    │
│  └────────────────┬───────────────────────────────────────┘    │
└───────────────────┼─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   REPORTING LAYER                               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Analytics Module                                       │    │
│  │  • Aggregate statistics (% plastic, polymer types)     │    │
│  │  • Temporal trends                                     │    │
│  │  • Spatial mapping                                     │    │
│  └────────────────┬───────────────────────────────────────┘    │
│                   │                                             │
│  ┌────────────────▼───────────────────────────────────────┐    │
│  │  Visualization Module                                   │    │
│  │  • Dashboard (real-time display)                       │    │
│  │  • Report generation (PDF/HTML)                        │    │
│  │  • Data export (CSV/JSON)                              │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Diagram

```
User/ROV System
     │
     ├─► [1] Upload Image + Metadata
     │         │
     │         ▼
     │   PreprocessingModule.process(image)
     │         │
     │         ├─► enhance_contrast(image)
     │         ├─► reduce_noise(image)
     │         ├─► extract_features(image)
     │         │         │
     │         │         ▼
     │         │   {features: dict}
     │         │
     │         ▼
     │   GeminiClassifier.classify(image, features, metadata)
     │         │
     │         ├─► build_prompt(image, features, metadata)
     │         ├─► call_gemini_api(prompt)
     │         │         │
     │         │         ▼ (Cloud API)
     │         │   Gemini 3 Pro Processing
     │         │         │
     │         │         ▼
     │         ├─► parse_response(json_response)
     │         ├─► validate_classification(result)
     │         │
     │         ▼
     │   {classification: dict}
     │         │
     │         ▼
     │   StorageModule.save(classification)
     │         │
     │         ├─► database.insert(record)
     │         ├─► cache.set(image_hash, result)
     │         │
     │         ▼
     │   ReportingModule.generate_summary()
     │         │
     │         ▼
     ├─◄ [2] Return Classification Result
     │
     └─► [3] View Dashboard/Reports
```

---

## 3. COMPONENT SPECIFICATIONS

### 3.1 Preprocessing Module

**File**: `preprocessing.py`

**Class**: `ImagePreprocessor`

**Methods**:

```python
class ImagePreprocessor:
    def __init__(self, config: dict):
        """Initialize preprocessor with configuration"""
        
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement"""
        
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for noise reduction"""
        
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1]"""
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline"""
```

**Class**: `FeatureExtractor`

**Methods**:

```python
class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor"""
        
    def detect_particles(self, image: np.ndarray) -> List[dict]:
        """Detect particles in image using contour detection"""
        
    def measure_size(self, contour: np.ndarray, scale: float) -> dict:
        """Measure particle size in micrometers"""
        
    def analyze_shape(self, contour: np.ndarray) -> dict:
        """Calculate shape descriptors (circularity, aspect ratio, etc.)"""
        
    def estimate_refractive_index(self, image: np.ndarray, particle_roi: np.ndarray) -> float:
        """Estimate RI from fringe spacing"""
        
    def extract_texture_features(self, image_roi: np.ndarray) -> dict:
        """Extract texture features (variance, entropy, etc.)"""
        
    def extract_all_features(self, image: np.ndarray, metadata: dict) -> List[dict]:
        """Extract all features for all detected particles"""
```

**Input Format**:
- `image`: NumPy array (H×W or H×W×3), uint8 or float32
- `metadata`: Dictionary with keys: `scale_um_per_pixel`, `water_temp`, `salinity`, etc.

**Output Format**:
```python
{
    'enhanced_image': np.ndarray,  # Preprocessed image
    'particles': [
        {
            'id': 1,
            'size_um': 250.5,
            'area_pixels': 1245,
            'perimeter_pixels': 145.2,
            'circularity': 0.68,
            'aspect_ratio': 2.3,
            'refractive_index_estimate': 1.57,
            'intensity_mean': 142.5,
            'intensity_variance': 18.3,
            'entropy': 4.2,
            'bounding_box': [x, y, w, h],
            'roi': np.ndarray  # Cropped particle region
        },
        # ... more particles
    ]
}
```

---

### 3.2 Classification Module

**File**: `classifier.py`

**Class**: `GeminiClassifier`

**Methods**:

```python
class GeminiClassifier:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """Initialize Gemini API client"""
        
    def build_prompt(self, particle_features: dict, image: np.ndarray, metadata: dict) -> str:
        """Construct multimodal prompt for Gemini"""
        
    def classify_particle(self, particle_features: dict, image: np.ndarray, metadata: dict) -> dict:
        """Classify single particle"""
        
    def classify_batch(self, particles: List[dict], image: np.ndarray, metadata: dict) -> List[dict]:
        """Classify multiple particles in batch"""
        
    def parse_response(self, response_text: str) -> dict:
        """Parse Gemini JSON response"""
        
    def validate_classification(self, classification: dict) -> bool:
        """Validate classification result structure"""
```

**Prompt Structure** (see separate Prompt Engineering Guide for details):

```python
prompt_template = """
You are a marine biology and materials science expert specializing in 
microplastic detection using holographic microscopy.

TASK: Classify this particle as MICROPLASTIC or BIOLOGICAL.

PARTICLE MEASUREMENTS:
- Size: {size_um} micrometers
- Circularity: {circularity}
- Aspect Ratio: {aspect_ratio}
- Estimated Refractive Index: {ri_estimate}
- Intensity Variance: {intensity_variance}

ENVIRONMENTAL CONTEXT:
- Water Temperature: {temp_c}°C
- Salinity: {salinity} ppt
- Depth: {depth_m} meters

[... domain knowledge framework ...]

RESPOND IN JSON:
{{
  "classification": "MICROPLASTIC | BIOLOGICAL | UNCERTAIN",
  "confidence_microplastic": 0-100,
  "confidence_biological": 0-100,
  "polymer_type": "PET | HDPE | PP | PS | PVC | Other | null",
  "organism_type": "diatom | copepod | other | null",
  "reasoning": "detailed explanation",
  "evidence": {{...}},
  "recommendation": "DEFINITE | PROBABLE | UNCERTAIN"
}}
"""
```

**Output Format**:
```python
{
    'particle_id': 1,
    'classification': 'MICROPLASTIC',  # or 'BIOLOGICAL', 'UNCERTAIN'
    'confidence_microplastic': 87,
    'confidence_biological': 10,
    'confidence_uncertain': 3,
    'polymer_type': 'PET',
    'polymer_confidence': 85,
    'organism_type': None,
    'recommendation': 'DEFINITE',  # or 'PROBABLE', 'UNCERTAIN'
    'reasoning': 'The regular, uniform diffraction pattern...',
    'evidence': {
        'diffraction_pattern': 'Regular, uniform fringes...',
        'morphology': 'Irregular, angular edges...',
        'color': 'Clear/translucent...',
        'size': '250μm consistent with microfiber...',
        'behavior': 'Static across frames...',
        'refractive_index': 'Estimated 1.57...'
    },
    'alternative_hypotheses': [
        'Could be mineral particle - but lacks crystalline pattern',
    ],
    'flags': [],  # or ['low_confidence', 'ambiguous_morphology', etc.]
    'processing_time_sec': 2.4,
    'timestamp': '2026-01-31T22:15:30Z'
}
```

---

### 3.3 Storage Module

**File**: `storage.py`

**Database Schema** (SQLite/PostgreSQL):

```sql
-- Main classifications table
CREATE TABLE classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL,
    particle_id INTEGER NOT NULL,
    classification TEXT NOT NULL,  -- 'MICROPLASTIC', 'BIOLOGICAL', 'UNCERTAIN'
    confidence_microplastic REAL,
    confidence_biological REAL,
    polymer_type TEXT,
    polymer_confidence REAL,
    organism_type TEXT,
    recommendation TEXT,
    reasoning TEXT,
    evidence JSON,
    alternative_hypotheses JSON,
    flags JSON,
    processing_time_sec REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(image_id, particle_id)
);

-- Particle features table
CREATE TABLE particle_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    classification_id INTEGER REFERENCES classifications(id),
    size_um REAL,
    circularity REAL,
    aspect_ratio REAL,
    refractive_index_estimate REAL,
    intensity_mean REAL,
    intensity_variance REAL,
    entropy REAL,
    bounding_box JSON,
    FOREIGN KEY(classification_id) REFERENCES classifications(id) ON DELETE CASCADE
);

-- Metadata table
CREATE TABLE metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL UNIQUE,
    capture_datetime DATETIME,
    location_lat REAL,
    location_lon REAL,
    depth_m REAL,
    water_temp_c REAL,
    salinity_ppt REAL,
    turbidity_ntu REAL,
    equipment_type TEXT,
    scale_um_per_pixel REAL
);

-- Aggregated statistics (for fast dashboard queries)
CREATE TABLE daily_statistics (
    date DATE PRIMARY KEY,
    total_particles INTEGER,
    microplastic_count INTEGER,
    biological_count INTEGER,
    uncertain_count INTEGER,
    microplastic_percentage REAL,
    polymer_distribution JSON,  -- {"PET": 45, "HDPE": 30, ...}
    avg_confidence REAL
);

-- Indexes for performance
CREATE INDEX idx_classifications_timestamp ON classifications(timestamp);
CREATE INDEX idx_classifications_classification ON classifications(classification);
CREATE INDEX idx_classifications_polymer ON classifications(polymer_type);
CREATE INDEX idx_metadata_datetime ON metadata(capture_datetime);
CREATE INDEX idx_metadata_location ON metadata(location_lat, location_lon);
```

**Class**: `StorageManager`

```python
class StorageManager:
    def __init__(self, db_path: str):
        """Initialize database connection"""
        
    def save_classification(self, classification: dict, features: dict, metadata: dict) -> int:
        """Save classification result to database"""
        
    def get_classification(self, image_id: str, particle_id: int) -> dict:
        """Retrieve specific classification"""
        
    def get_statistics(self, start_date: str, end_date: str) -> dict:
        """Calculate aggregated statistics for date range"""
        
    def export_csv(self, query_params: dict, output_path: str):
        """Export filtered data to CSV"""
        
    def update_daily_statistics(self, date: str):
        """Update aggregated statistics table"""
```

---

### 3.4 Reporting Module

**File**: `reporting.py`

**Class**: `ReportGenerator`

```python
class ReportGenerator:
    def __init__(self, storage: StorageManager):
        """Initialize report generator"""
        
    def generate_summary_report(self, start_date: str, end_date: str) -> dict:
        """Generate summary statistics"""
        
    def create_dashboard_data(self) -> dict:
        """Generate data for real-time dashboard"""
        
    def export_pdf_report(self, data: dict, output_path: str):
        """Generate PDF report with charts"""
        
    def create_confusion_matrix(self, ground_truth: List, predictions: List) -> np.ndarray:
        """Calculate confusion matrix for validation"""
```

**Report Output Structure**:

```python
{
    'period': {
        'start_date': '2026-01-01',
        'end_date': '2026-01-31'
    },
    'summary': {
        'total_particles': 1245,
        'microplastics': 892,
        'biological': 301,
        'uncertain': 52,
        'microplastic_percentage': 71.6
    },
    'polymer_distribution': {
        'PET': 402,
        'HDPE': 198,
        'PP': 165,
        'PS': 89,
        'PVC': 38
    },
    'confidence_stats': {
        'mean_confidence': 82.4,
        'high_confidence_count': 1104,  # >85%
        'medium_confidence_count': 89,   # 60-85%
        'low_confidence_count': 52       # <60%
    },
    'temporal_trends': [
        {'date': '2026-01-01', 'count': 42, 'percentage': 68.5},
        # ... daily data
    ],
    'spatial_distribution': [
        {'location': 'Site A', 'lat': 34.02, 'lon': -118.45, 'count': 345},
        # ... location data
    ],
    'quality_metrics': {
        'avg_processing_time_sec': 2.8,
        'api_success_rate': 99.2
    }
}
```

---

## 4. DATA PIPELINE

### 4.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA INGESTION                                     │
│                                                             │
│ Input: Raw holographic image (PNG/TIFF) + metadata (JSON)  │
│   └─► Validation (format, size, required fields)           │
│       └─► Store original in archive                        │
│           └─► Generate unique image_id (SHA-256 hash)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: PREPROCESSING                                      │
│                                                             │
│ Step 2.1: Image Enhancement                                │
│   └─► Load image as NumPy array                            │
│       └─► Convert to grayscale (if color)                  │
│           └─► CLAHE contrast enhancement                   │
│               └─► Bilateral noise reduction                │
│                   └─► Normalize to [0, 1]                  │
│                                                             │
│ Step 2.2: Feature Extraction                               │
│   └─► Particle detection (thresholding + contours)         │
│       └─► For each particle:                               │
│           ├─► Measure size (area → micrometers)            │
│           ├─► Calculate shape features                     │
│           ├─► Estimate refractive index                    │
│           ├─► Extract texture features                     │
│           └─► Crop ROI for Gemini                          │
│                                                             │
│ Output: Enhanced image + feature vector per particle       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: CLASSIFICATION (PER PARTICLE)                     │
│                                                             │
│ Step 3.1: Prompt Construction                              │
│   └─► Embed particle ROI image                             │
│       └─► Format feature vector as text                    │
│           └─► Add domain knowledge context                 │
│               └─► Include environmental metadata           │
│                                                             │
│ Step 3.2: Gemini API Call                                  │
│   └─► Send multimodal request                              │
│       └─► Wait for response (timeout: 30s)                 │
│           └─► Retry on failure (max 3 attempts)            │
│                                                             │
│ Step 3.3: Response Processing                              │
│   └─► Parse JSON response                                  │
│       └─► Validate required fields                         │
│           └─► Extract classification + confidence          │
│               └─► Flag if confidence <60%                  │
│                                                             │
│ Output: Classification dict per particle                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: STORAGE                                            │
│                                                             │
│ Step 4.1: Database Insert                                  │
│   └─► Insert into classifications table                    │
│       └─► Insert into particle_features table              │
│           └─► Update metadata table                        │
│                                                             │
│ Step 4.2: Caching (Optional)                               │
│   └─► Cache result by image_hash + particle_id             │
│       └─► TTL: 24 hours                                    │
│                                                             │
│ Step 4.3: Update Aggregates                                │
│   └─► Increment daily statistics counters                  │
│       └─► Update polymer distribution                      │
│                                                             │
│ Output: Database record ID                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 5: REPORTING (ASYNC/BATCH)                           │
│                                                             │
│ Step 5.1: Real-Time Dashboard Update                       │
│   └─► Push new classification to dashboard                 │
│       └─► Update live statistics counters                  │
│                                                             │
│ Step 5.2: Batch Report Generation (Daily)                  │
│   └─► Query daily statistics                               │
│       └─► Generate charts (temporal trends, distributions) │
│           └─► Export PDF report                            │
│                                                             │
│ Output: Dashboard + PDF reports                            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Error Handling Strategy

**Error Types and Responses:**

| Error Type | Response | Fallback |
|-----------|----------|----------|
| Image load failure | Log error, skip image | Continue to next |
| Preprocessing crash | Log error, use raw image | Warn user |
| No particles detected | Log warning, return empty | Skip classification |
| Gemini API timeout | Retry up to 3 times | Flag as "processing_failed" |
| Gemini API rate limit | Exponential backoff (1s, 2s, 4s) | Queue for later |
| JSON parse error | Log response, manual review | Flag as "parse_error" |
| Database insert failure | Log error, retry once | Write to backup file |
| Low confidence (<40%) | Flag for manual review | Store as "UNCERTAIN" |

**Logging Levels:**
- **DEBUG**: Feature extraction details, API request/response
- **INFO**: Successful classifications, batch processing progress
- **WARNING**: Low confidence classifications, retry attempts
- **ERROR**: API failures, database errors, crashes
- **CRITICAL**: System-level failures (out of memory, disk full)

---

## 5. API INTEGRATION

### 5.1 Gemini API Configuration

**Authentication:**
```python
import google.generativeai as genai

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-pro')
```

**API Quotas and Limits:**
- Free tier: 1,000,000 tokens/day
- Rate limit: 60 requests/minute
- Max input size: 10MB per request
- Timeout: 30 seconds default

**Request Structure:**
```python
response = model.generate_content([
    {
        'mime_type': 'image/png',
        'data': base64_encoded_image
    },
    prompt_text
])
```

**Response Handling:**
```python
try:
    result = json.loads(response.text)
    if validate_response(result):
        return result
    else:
        raise ValidationError("Invalid response structure")
except json.JSONDecodeError:
    log.error(f"Failed to parse JSON: {response.text}")
    return None
except Exception as e:
    log.error(f"Gemini API error: {e}")
    return None
```

### 5.2 Rate Limiting and Retry Logic

**Exponential Backoff:**
```python
import time
from typing import Optional

def call_gemini_with_retry(prompt: str, image: bytes, max_retries: int = 3) -> Optional[dict]:
    for attempt in range(max_retries):
        try:
            response = model.generate_content([image, prompt])
            return parse_response(response.text)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                log.warning(f"Retry {attempt+1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                log.error(f"Max retries reached: {e}")
                return None
```

**Token Management:**
```python
def estimate_token_count(image_size_bytes: int, prompt_length: int) -> int:
    """Rough estimate: 1 token ≈ 4 chars, image ≈ size/10 tokens"""
    return prompt_length // 4 + image_size_bytes // 10

def check_daily_quota(storage: StorageManager) -> bool:
    """Check if we've exceeded daily quota"""
    today_count = storage.get_daily_token_usage()
    return today_count < 900_000  # Leave buffer under 1M limit
```

---

## 6. SECURITY & PRIVACY

### 6.1 API Key Management

**Best Practices:**
- Store API key in environment variable (never hardcode)
- Use `.env` file for local development (add to `.gitignore`)
- Use secrets manager for production (GCP Secret Manager, AWS Secrets Manager)

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
```

### 6.2 Data Privacy

**Image Handling:**
- Images sent to Gemini API may be retained by Google (per API terms)
- For sensitive data: Consider on-premise fine-tuned models (future)
- Anonymize location data if required by regulations

**Database Security:**
- Encrypt database at rest (SQLCipher for SQLite, native encryption for PostgreSQL)
- Restrict database access (firewall rules, authentication)
- Regular backups (automated, encrypted)

---

## 7. PERFORMANCE REQUIREMENTS

### 7.1 Latency Targets

| Operation | Target | Acceptable | Notes |
|-----------|--------|-----------|-------|
| Image preprocessing | <1s | <2s | Dependent on image size |
| Feature extraction | <0.5s | <1s | Per particle |
| Gemini API call | <3s | <10s | Network dependent |
| Database insert | <0.1s | <0.5s | Per classification |
| **Total per particle** | **<5s** | **<10s** | End-to-end |

### 7.2 Throughput Targets

- **Single-threaded**: 10-15 particles/minute
- **Multi-threaded (4 cores)**: 40-60 particles/minute
- **Batch processing**: 500-1000 particles/hour

### 7.3 Resource Requirements

**Minimum Hardware:**
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Disk: 10 GB (for database + logs)
- Network: 5 Mbps

**Recommended Hardware:**
- CPU: 4+ cores, 3.0 GHz
- RAM: 8 GB
- Disk: 50 GB SSD
- Network: 10+ Mbps

---

## 8. SCALABILITY DESIGN

### 8.1 Horizontal Scaling

**Parallel Processing Architecture:**

```
                    Load Balancer
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   Worker 1          Worker 2          Worker 3
   (Instance)        (Instance)        (Instance)
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
                 Shared Database
              (PostgreSQL/Cloud SQL)
```

**Queue-Based Processing:**

```
Image Upload → Redis Queue → Worker Pool → Database
                  │              │
                  │              └─► Gemini API (parallel requests)
                  │
                  └─► Dashboard (real-time updates)
```

### 8.2 Caching Strategy

**Multi-Level Caching:**

1. **L1: In-Memory (Python dict)**
   - Recent classifications (last 100)
   - TTL: 5 minutes
   - Eviction: LRU

2. **L2: Redis (optional)**
   - Recent classifications (last 10,000)
   - TTL: 24 hours
   - Key: `classification:{image_hash}:{particle_id}`

3. **L3: Database**
   - All historical data
   - Indexed queries for fast retrieval

---

## 9. DEPLOYMENT ARCHITECTURE

### 9.1 Docker Containerization

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for API (if using FastAPI)
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

### 9.2 Cloud Deployment (Google Cloud Run)

**Configuration:**
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: micro-scale-sentinel
spec:
  template:
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/micro-scale-sentinel:latest
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: gemini-api-key
              key: key
        resources:
          limits:
            memory: 2Gi
            cpu: 2
```

---

## 10. MONITORING & OBSERVABILITY

### 10.1 Key Metrics

**System Metrics:**
- Request rate (classifications/minute)
- Error rate (failed classifications/total)
- Latency (p50, p95, p99)
- API quota usage

**Business Metrics:**
- Microplastic detection rate
- Confidence score distribution
- Polymer type distribution
- Uncertain classification rate

### 10.2 Logging

**Structured Logging Format:**
```python
{
    'timestamp': '2026-01-31T22:15:30Z',
    'level': 'INFO',
    'component': 'classifier',
    'action': 'classification_complete',
    'image_id': 'abc123',
    'particle_id': 1,
    'classification': 'MICROPLASTIC',
    'confidence': 87,
    'processing_time_sec': 2.4
}
```

---

## APPENDIX A: Configuration File Format

**config.yaml:**
```yaml
system:
  name: "Micro-Scale Sentinel"
  version: "1.0.0"

gemini:
  model: "gemini-1.5-pro"
  timeout_sec: 30
  max_retries: 3
  
preprocessing:
  clahe:
    clip_limit: 3.0
    tile_grid_size: [8, 8]
  bilateral:
    d: 9
    sigma_color: 75
    sigma_space: 75
    
feature_extraction:
  scale_um_per_pixel: 0.5
  min_particle_size_um: 50
  max_particle_size_um: 5000
  
classification:
  confidence_threshold_high: 85
  confidence_threshold_low: 60
  
storage:
  database_path: "data/sentinel.db"
  backup_enabled: true
  backup_interval_hours: 24
  
reporting:
  dashboard_update_interval_sec: 5
  daily_report_time: "00:00"
```

---

**Document Version**: 1.0
**Last Updated**: January 31, 2026
**Author**: Micro-Scale Sentinel Team
**Status**: Production Ready