# Micro-Scale Sentinel: Data Format Specifications
## Version 1.0 - January 31, 2026

---

## Table of Contents

1. [Input Data Formats](#1-input-data-formats)
2. [Intermediate Data Formats](#2-intermediate-data-formats)
3. [Output Data Formats](#3-output-data-formats)
4. [Database Schema](#4-database-schema)
5. [File Formats & Standards](#5-file-formats--standards)
6. [API Specifications](#6-api-specifications)

---

## 1. INPUT DATA FORMATS

### 1.1 Holographic Image Format

**Primary Input: Raw Holographic Microscopy Image**

```
Format Requirements:
├─ File Format: PNG, TIFF, or JPEG
├─ Bit Depth: 8-bit (grayscale preferred) or 16-bit
├─ Color Space: Grayscale or RGB (will be converted to grayscale)
├─ Resolution: 256×256 to 2048×2048 pixels (512×512 recommended)
├─ Compression: Lossless (PNG/TIFF) preferred; JPEG acceptable
└─ File Size: <10 MB per image
```

**Image Naming Convention:**
```
{site_id}_{date}_{time}_{sample_id}_{particle_id}.png

Example: 
SITE001_20260131_143052_S042_P001.png

Where:
- SITE001 = Location identifier
- 20260131 = Date (YYYYMMDD)
- 143052 = Time (HHMMSS)
- S042 = Sample number
- P001 = Particle number within sample
```

**Image Quality Requirements:**
- Signal-to-noise ratio (SNR): >10 dB
- Contrast: Sufficient to distinguish particle from background
- Focus: Particles should be in the focal plane
- Illumination: Even, no significant vignetting

---

### 1.2 Metadata Format (JSON)

**Companion Metadata File**

Each holographic image should have an accompanying JSON metadata file with the same base name:

```json
{
  "image_metadata": {
    "file_name": "SITE001_20260131_143052_S042_P001.png",
    "image_id": "a3f8d2e1b4c5",
    "capture_datetime": "2026-01-31T14:30:52Z",
    "format": "PNG",
    "resolution": {
      "width_px": 512,
      "height_px": 512
    },
    "bit_depth": 8,
    "color_space": "grayscale"
  },
  
  "microscope_settings": {
    "equipment_type": "DHM (Digital Holographic Microscope)",
    "manufacturer": "Lyncée Tec / 4Deep / Custom",
    "model": "DHM R2100",
    "magnification": "20x",
    "wavelength_nm": 632.8,
    "numerical_aperture": 0.4,
    "scale_um_per_pixel": 0.5,
    "reconstruction_distance_mm": 15.2
  },
  
  "environmental_conditions": {
    "location": {
      "site_id": "SITE001",
      "site_name": "Santa Monica Bay Sampling Station A",
      "latitude": 34.0195,
      "longitude": -118.4912,
      "depth_m": 15.5
    },
    "water_properties": {
      "temperature_c": 16.2,
      "salinity_ppt": 33.8,
      "pH": 8.1,
      "turbidity_ntu": 2.4,
      "dissolved_oxygen_mg_per_l": 7.8
    },
    "sampling": {
      "sample_id": "S042",
      "sample_volume_ml": 500,
      "filtration_pore_size_um": 50,
      "collection_method": "Plankton net tow",
      "collection_depth_m": 10.0
    }
  },
  
  "particle_metadata": {
    "particle_id": "P001",
    "detection_method": "automated",
    "detection_confidence": 0.95,
    "frame_number": 1,
    "total_frames": 30
  },
  
  "processing_metadata": {
    "preprocessing_version": "1.0.0",
    "preprocessing_timestamp": "2026-01-31T14:35:10Z",
    "preprocessing_parameters": {
      "clahe_clip_limit": 3.0,
      "bilateral_d": 9,
      "bilateral_sigma_color": 75,
      "bilateral_sigma_space": 75
    }
  }
}
```

**Required Fields** (Minimum):
- `image_metadata.file_name`
- `image_metadata.capture_datetime`
- `microscope_settings.scale_um_per_pixel`
- `environmental_conditions.location.latitude`
- `environmental_conditions.location.longitude`

**Optional Fields**:
All other fields are optional but recommended for better classification accuracy and scientific reproducibility.

---

### 1.3 Batch Input Format (CSV)

For batch processing, a CSV file can be provided listing multiple images and their metadata:

```csv
image_file,capture_datetime,site_id,latitude,longitude,depth_m,water_temp_c,salinity_ppt,scale_um_per_pixel
SITE001_20260131_143052_S042_P001.png,2026-01-31T14:30:52Z,SITE001,34.0195,-118.4912,15.5,16.2,33.8,0.5
SITE001_20260131_143053_S042_P002.png,2026-01-31T14:30:53Z,SITE001,34.0195,-118.4912,15.5,16.2,33.8,0.5
SITE001_20260131_143054_S042_P003.png,2026-01-31T14:30:54Z,SITE001,34.0195,-118.4912,15.5,16.2,33.8,0.5
```

**CSV Format Requirements:**
- Encoding: UTF-8
- Delimiter: Comma (`,`)
- Header row: Required
- Date/time format: ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`)
- Decimal separator: Period (`.`)
- Missing values: Empty field or `null`

---

## 2. INTERMEDIATE DATA FORMATS

### 2.1 Preprocessed Image Format

After preprocessing, images are stored as NumPy arrays:

```python
# Preprocessed image dictionary
preprocessed_data = {
    'original_image': np.ndarray,      # Shape: (H, W), dtype: uint8
    'enhanced_image': np.ndarray,      # Shape: (H, W), dtype: float32, range: [0, 1]
    'preprocessing_applied': [
        'grayscale_conversion',
        'clahe_enhancement',
        'bilateral_filtering',
        'normalization'
    ],
    'preprocessing_params': {
        'clahe_clip_limit': 3.0,
        'bilateral_d': 9,
        'bilateral_sigma_color': 75,
        'bilateral_sigma_space': 75
    }
}
```

### 2.2 Feature Vector Format

Extracted features for each particle:

```python
# Feature dictionary per particle
particle_features = {
    # Identification
    'particle_id': 'P001',
    'image_id': 'a3f8d2e1b4c5',
    
    # Geometric features
    'size_um': 245.3,                    # Equivalent diameter
    'area_um2': 47234.2,                 # Particle area
    'perimeter_um': 789.5,               # Particle perimeter
    'circularity': 0.68,                 # 4π × area / perimeter²
    'aspect_ratio': 2.4,                 # Major axis / minor axis
    'solidity': 0.89,                    # Area / convex hull area
    'extent': 0.72,                      # Area / bounding box area
    
    # Optical features
    'refractive_index_estimate': 1.57,   # From fringe spacing
    'refractive_index_std': 0.02,        # Variation across particle
    'fringe_spacing_um': 2.3,            # Average fringe spacing
    'fringe_count': 12,                  # Number of visible fringes
    
    # Intensity features
    'intensity_mean': 142.5,             # Mean grayscale value (0-255)
    'intensity_std': 18.3,               # Standard deviation
    'intensity_min': 98,                 # Minimum value
    'intensity_max': 201,                # Maximum value
    'intensity_range': 103,              # Max - min
    'intensity_variance': 334.89,        # Variance
    
    # Texture features
    'entropy': 4.23,                     # Shannon entropy
    'contrast': 0.45,                    # Local contrast
    'homogeneity': 0.82,                 # Grayscale uniformity
    'energy': 0.31,                      # Uniformity of distribution
    
    # Spatial features
    'centroid_x': 256.5,                 # Particle center X (pixels)
    'centroid_y': 189.3,                 # Particle center Y (pixels)
    'bounding_box': {
        'x': 210,
        'y': 145,
        'width': 93,
        'height': 89
    },
    
    # Region of interest
    'roi_image': np.ndarray,             # Cropped particle image
    
    # Quality metrics
    'snr_db': 15.3,                      # Signal-to-noise ratio
    'focus_measure': 0.87,               # Focus quality (0-1)
    'edge_sharpness': 0.72               # Edge clarity metric
}
```

**Feature Vector as NumPy Array** (for ML if needed later):

```python
# Flattened feature vector for ML algorithms
feature_vector = np.array([
    245.3,      # size_um
    0.68,       # circularity
    2.4,        # aspect_ratio
    1.57,       # refractive_index_estimate
    142.5,      # intensity_mean
    18.3,       # intensity_std
    4.23,       # entropy
    0.45,       # contrast
    # ... other features
])
```

---

## 3. OUTPUT DATA FORMATS

### 3.1 Classification Result Format (JSON)

**Single Particle Classification**

```json
{
  "metadata": {
    "image_id": "a3f8d2e1b4c5",
    "particle_id": "P001",
    "classification_id": "CLS_20260131_143710_001",
    "timestamp": "2026-01-31T14:37:10Z",
    "processing_time_sec": 2.4,
    "system_version": "1.0.0"
  },
  
  "classification": {
    "primary_classification": "MICROPLASTIC",
    "confidence_microplastic": 87,
    "confidence_biological": 10,
    "confidence_uncertain": 3,
    "recommendation": "DEFINITE"
  },
  
  "polymer_identification": {
    "polymer_type": "PET",
    "polymer_confidence": 85,
    "polymer_source_likely": "synthetic_textile",
    "alternative_polymers": [
      {"type": "PS", "confidence": 10},
      {"type": "HDPE", "confidence": 5}
    ]
  },
  
  "biological_identification": {
    "organism_type": null,
    "organism_confidence": null,
    "alternative_organisms": []
  },
  
  "evidence": {
    "diffraction_pattern": "Regular, uniform fringes at 2.3μm spacing consistent with RI=1.575 (PET)",
    "refractive_index_analysis": "Estimated RI of 1.57 matches PET polymer (expected 1.575)",
    "morphology": "Irregular, angular edges typical of fragmented polymer fiber",
    "color_transparency": "Clear/translucent, no biological pigmentation observed",
    "size_analysis": "250μm length consistent with microfiber fragment from textiles",
    "behavior": "Static across frame sequence; no movement or rotation detected"
  },
  
  "reasoning": {
    "summary": "The regular, uniform diffraction pattern with spacing consistent with RI=1.57 strongly indicates PET plastic. The irregular, angular edges are consistent with mechanical fragmentation from weathering. The absence of cellular structures, pigmentation, or movement rules out biological origin.",
    "hypothesis_ranking": [
      {
        "hypothesis": "PET microfiber from synthetic textile",
        "confidence": 87,
        "supporting_evidence": [
          "Refractive index 1.57 matches PET",
          "Elongated fiber morphology (aspect ratio 2.4)",
          "Uniform transparency",
          "No cellular structure visible"
        ],
        "contradicting_evidence": []
      },
      {
        "hypothesis": "Polystyrene fragment",
        "confidence": 10,
        "supporting_evidence": ["RI in similar range"],
        "contradicting_evidence": ["Morphology inconsistent with PS brittleness"]
      }
    ]
  },
  
  "alternative_hypotheses": [
    "Could be mineral particle (e.g., silicate) - but lacks crystalline diffraction pattern",
    "Could be degraded cellulose - but RI and morphology inconsistent with natural fibers"
  ],
  
  "quality_flags": {
    "flags": [],
    "warnings": [],
    "needs_manual_review": false,
    "review_priority": "low"
  },
  
  "particle_features": {
    "size_um": 250.5,
    "circularity": 0.68,
    "aspect_ratio": 2.4,
    "refractive_index_estimate": 1.57,
    "intensity_mean": 142.5,
    "entropy": 4.2
  },
  
  "environmental_context": {
    "location": "Santa Monica Bay Station A",
    "latitude": 34.0195,
    "longitude": -118.4912,
    "depth_m": 15.5,
    "water_temp_c": 16.2,
    "salinity_ppt": 33.8
  }
}
```

### 3.2 Batch Classification Results (JSON Array)

```json
{
  "batch_metadata": {
    "batch_id": "BATCH_20260131_001",
    "site_id": "SITE001",
    "sample_id": "S042",
    "total_particles": 125,
    "processed_successfully": 123,
    "processing_failed": 2,
    "start_time": "2026-01-31T14:30:00Z",
    "end_time": "2026-01-31T14:45:32Z",
    "total_processing_time_sec": 932
  },
  
  "summary_statistics": {
    "microplastic_count": 89,
    "biological_count": 31,
    "uncertain_count": 3,
    "microplastic_percentage": 72.4,
    "polymer_distribution": {
      "PET": 42,
      "HDPE": 21,
      "PP": 15,
      "PS": 8,
      "PVC": 3
    },
    "mean_confidence": 84.2,
    "high_confidence_count": 108,
    "low_confidence_count": 15
  },
  
  "classifications": [
    {
      "particle_id": "P001",
      "classification": "MICROPLASTIC",
      "polymer_type": "PET",
      "confidence": 87
    },
    {
      "particle_id": "P002",
      "classification": "BIOLOGICAL",
      "organism_type": "diatom",
      "confidence": 91
    }
    // ... more classifications
  ],
  
  "failed_classifications": [
    {
      "particle_id": "P078",
      "error": "API timeout",
      "retry_attempted": true
    },
    {
      "particle_id": "P112",
      "error": "Invalid image format",
      "retry_attempted": false
    }
  ]
}
```

### 3.3 Export Formats

**CSV Export (For Spreadsheet Analysis)**

```csv
particle_id,image_id,classification,confidence,polymer_type,organism_type,size_um,circularity,ri_estimate,latitude,longitude,capture_datetime
P001,a3f8d2e1b4c5,MICROPLASTIC,87,PET,,250.5,0.68,1.57,34.0195,-118.4912,2026-01-31T14:30:52Z
P002,b2c4e5f1a3d6,BIOLOGICAL,91,,diatom,145.2,0.89,1.38,34.0195,-118.4912,2026-01-31T14:30:53Z
```

**GeoJSON Export (For Mapping)**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-118.4912, 34.0195]
      },
      "properties": {
        "particle_id": "P001",
        "classification": "MICROPLASTIC",
        "polymer_type": "PET",
        "confidence": 87,
        "size_um": 250.5,
        "capture_datetime": "2026-01-31T14:30:52Z"
      }
    }
  ]
}
```

---

## 4. DATABASE SCHEMA

### 4.1 Complete SQL Schema

```sql
-- Main classifications table
CREATE TABLE classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    classification_id TEXT UNIQUE NOT NULL,
    image_id TEXT NOT NULL,
    particle_id TEXT NOT NULL,
    
    -- Classification results
    primary_classification TEXT NOT NULL CHECK(primary_classification IN ('MICROPLASTIC', 'BIOLOGICAL', 'UNCERTAIN')),
    confidence_microplastic REAL CHECK(confidence_microplastic BETWEEN 0 AND 100),
    confidence_biological REAL CHECK(confidence_biological BETWEEN 0 AND 100),
    confidence_uncertain REAL CHECK(confidence_uncertain BETWEEN 0 AND 100),
    recommendation TEXT CHECK(recommendation IN ('DEFINITE', 'PROBABLE', 'UNCERTAIN')),
    
    -- Polymer identification
    polymer_type TEXT CHECK(polymer_type IN ('PET', 'HDPE', 'PP', 'PS', 'PVC', 'Other', NULL)),
    polymer_confidence REAL CHECK(polymer_confidence BETWEEN 0 AND 100 OR polymer_confidence IS NULL),
    polymer_source_likely TEXT,
    
    -- Biological identification
    organism_type TEXT,
    organism_confidence REAL CHECK(organism_confidence BETWEEN 0 AND 100 OR organism_confidence IS NULL),
    
    -- Evidence and reasoning
    reasoning TEXT NOT NULL,
    evidence_json TEXT,  -- JSON blob
    alternative_hypotheses_json TEXT,  -- JSON array
    
    -- Quality metrics
    needs_manual_review BOOLEAN DEFAULT FALSE,
    review_priority TEXT CHECK(review_priority IN ('low', 'medium', 'high')),
    flags_json TEXT,  -- JSON array
    
    -- Processing metadata
    processing_time_sec REAL,
    system_version TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint
    UNIQUE(image_id, particle_id)
);

-- Particle features table
CREATE TABLE particle_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    classification_id INTEGER NOT NULL,
    
    -- Geometric features
    size_um REAL,
    area_um2 REAL,
    perimeter_um REAL,
    circularity REAL CHECK(circularity BETWEEN 0 AND 1),
    aspect_ratio REAL,
    solidity REAL CHECK(solidity BETWEEN 0 AND 1),
    extent REAL CHECK(extent BETWEEN 0 AND 1),
    
    -- Optical features
    refractive_index_estimate REAL,
    refractive_index_std REAL,
    fringe_spacing_um REAL,
    fringe_count INTEGER,
    
    -- Intensity features
    intensity_mean REAL,
    intensity_std REAL,
    intensity_min REAL,
    intensity_max REAL,
    intensity_variance REAL,
    
    -- Texture features
    entropy REAL,
    contrast REAL,
    homogeneity REAL,
    energy REAL,
    
    -- Spatial features
    centroid_x REAL,
    centroid_y REAL,
    bounding_box_json TEXT,  -- JSON object
    
    -- Quality metrics
    snr_db REAL,
    focus_measure REAL CHECK(focus_measure BETWEEN 0 AND 1),
    edge_sharpness REAL CHECK(edge_sharpness BETWEEN 0 AND 1),
    
    FOREIGN KEY(classification_id) REFERENCES classifications(id) ON DELETE CASCADE
);

-- Metadata table
CREATE TABLE image_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    
    -- Capture information
    capture_datetime DATETIME NOT NULL,
    
    -- Location
    site_id TEXT,
    site_name TEXT,
    latitude REAL CHECK(latitude BETWEEN -90 AND 90),
    longitude REAL CHECK(longitude BETWEEN -180 AND 180),
    depth_m REAL CHECK(depth_m >= 0),
    
    -- Environmental conditions
    water_temp_c REAL,
    salinity_ppt REAL CHECK(salinity_ppt >= 0),
    ph REAL CHECK(ph BETWEEN 0 AND 14),
    turbidity_ntu REAL CHECK(turbidity_ntu >= 0),
    dissolved_oxygen_mg_per_l REAL CHECK(dissolved_oxygen_mg_per_l >= 0),
    
    -- Sampling information
    sample_id TEXT,
    sample_volume_ml REAL,
    filtration_pore_size_um REAL,
    collection_method TEXT,
    collection_depth_m REAL,
    
    -- Microscope settings
    equipment_type TEXT,
    manufacturer TEXT,
    model TEXT,
    magnification TEXT,
    wavelength_nm REAL,
    numerical_aperture REAL,
    scale_um_per_pixel REAL NOT NULL,
    
    -- Image properties
    resolution_width_px INTEGER,
    resolution_height_px INTEGER,
    bit_depth INTEGER,
    color_space TEXT
);

-- Aggregated statistics table (for fast dashboard queries)
CREATE TABLE daily_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    site_id TEXT,
    
    -- Counts
    total_particles INTEGER DEFAULT 0,
    microplastic_count INTEGER DEFAULT 0,
    biological_count INTEGER DEFAULT 0,
    uncertain_count INTEGER DEFAULT 0,
    
    -- Percentages
    microplastic_percentage REAL,
    
    -- Polymer distribution
    pet_count INTEGER DEFAULT 0,
    hdpe_count INTEGER DEFAULT 0,
    pp_count INTEGER DEFAULT 0,
    ps_count INTEGER DEFAULT 0,
    pvc_count INTEGER DEFAULT 0,
    other_polymer_count INTEGER DEFAULT 0,
    
    -- Quality metrics
    avg_confidence REAL,
    high_confidence_count INTEGER DEFAULT 0,
    low_confidence_count INTEGER DEFAULT 0,
    
    -- Processing metrics
    total_processing_time_sec REAL,
    avg_processing_time_sec REAL,
    
    UNIQUE(date, site_id)
);

-- Indexes for performance
CREATE INDEX idx_classifications_timestamp ON classifications(timestamp);
CREATE INDEX idx_classifications_classification ON classifications(primary_classification);
CREATE INDEX idx_classifications_polymer ON classifications(polymer_type);
CREATE INDEX idx_classifications_confidence ON classifications(confidence_microplastic);
CREATE INDEX idx_classifications_image ON classifications(image_id);
CREATE INDEX idx_metadata_datetime ON image_metadata(capture_datetime);
CREATE INDEX idx_metadata_site ON image_metadata(site_id);
CREATE INDEX idx_metadata_location ON image_metadata(latitude, longitude);
CREATE INDEX idx_daily_stats_date ON daily_statistics(date);
CREATE INDEX idx_daily_stats_site ON daily_statistics(site_id);
```

---

## 5. FILE FORMATS & STANDARDS

### 5.1 Directory Structure

```
project_root/
│
├── data/
│   ├── raw_images/
│   │   ├── SITE001/
│   │   │   ├── 20260131/
│   │   │   │   ├── SITE001_20260131_143052_S042_P001.png
│   │   │   │   ├── SITE001_20260131_143052_S042_P001_metadata.json
│   │   │   │   ├── SITE001_20260131_143053_S042_P002.png
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── preprocessed/
│   │   ├── SITE001/
│   │   │   └── 20260131/
│   │   │       ├── SITE001_20260131_143052_S042_P001_enhanced.npy
│   │   │       └── ...
│   │   └── ...
│   │
│   ├── results/
│   │   ├── classifications/
│   │   │   ├── 20260131/
│   │   │   │   ├── batch_SITE001_20260131.json
│   │   │   │   └── ...
│   │   │   └── ...
│   │   │
│   │   ├── exports/
│   │   │   ├── 20260131_microplastics.csv
│   │   │   ├── 20260131_spatial.geojson
│   │   │   └── ...
│   │   │
│   │   └── reports/
│   │       ├── daily/
│   │       │   ├── report_20260131.pdf
│   │       │   └── ...
│   │       └── monthly/
│   │           └── report_202601.pdf
│   │
│   └── database/
│       └── sentinel.db
│
├── config/
│   ├── config.yaml
│   ├── .env (API keys - not committed to git)
│   └── sites.json (site definitions)
│
└── logs/
    ├── 20260131.log
    └── ...
```

### 5.2 Configuration File Format

**config.yaml**

```yaml
# System Configuration
system:
  name: "Micro-Scale Sentinel"
  version: "1.0.0"
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Gemini API Configuration
gemini:
  model: "gemini-1.5-pro"
  api_key_env_var: "GEMINI_API_KEY"
  timeout_sec: 30
  max_retries: 3
  rate_limit_requests_per_minute: 60

# Preprocessing Configuration
preprocessing:
  clahe:
    enabled: true
    clip_limit: 3.0
    tile_grid_size: [8, 8]
  bilateral:
    enabled: true
    d: 9
    sigma_color: 75
    sigma_space: 75
  normalization:
    enabled: true
    method: "min_max"  # "min_max" or "z_score"

# Feature Extraction Configuration
feature_extraction:
  scale_um_per_pixel: 0.5  # Default if not in metadata
  min_particle_size_um: 50
  max_particle_size_um: 5000
  detection_threshold: 0.5  # Confidence threshold for particle detection

# Classification Configuration
classification:
  confidence_threshold_high: 85  # DEFINITE recommendation
  confidence_threshold_low: 60   # Below = UNCERTAIN recommendation
  use_full_prompt: true  # false = use simplified prompt
  enable_verification: false  # Re-analyze uncertain cases

# Storage Configuration
storage:
  database_type: "sqlite"  # "sqlite" or "postgresql"
  database_path: "data/database/sentinel.db"  # For SQLite
  # For PostgreSQL:
  # database_host: "localhost"
  # database_port: 5432
  # database_name: "microplastics"
  # database_user: "sentinel"
  # database_password_env_var: "DB_PASSWORD"
  backup_enabled: true
  backup_interval_hours: 24
  backup_location: "backups/"

# Reporting Configuration
reporting:
  dashboard_enabled: true
  dashboard_update_interval_sec: 5
  daily_report_enabled: true
  daily_report_time: "00:00"  # HH:MM (24-hour)
  export_formats: ["csv", "json", "geojson"]

# Paths
paths:
  raw_images: "data/raw_images/"
  preprocessed: "data/preprocessed/"
  results: "data/results/"
  exports: "data/results/exports/"
  reports: "data/results/reports/"
  logs: "logs/"
```

---

## 6. API SPECIFICATIONS

### 6.1 REST API Endpoints (If Deployed as Service)

**Base URL**: `https://api.microscale-sentinel.org/v1`

#### **POST /classify/single**

Classify a single particle

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "metadata": {
    "scale_um_per_pixel": 0.5,
    "water_temp_c": 16.2,
    "salinity_ppt": 33.8,
    "latitude": 34.0195,
    "longitude": -118.4912
  }
}
```

**Response:**
```json
{
  "status": "success",
  "classification": {
    "primary_classification": "MICROPLASTIC",
    "confidence": 87,
    "polymer_type": "PET",
    "reasoning": "..."
  }
}
```

#### **POST /classify/batch**

Classify multiple particles

**Request:**
```json
{
  "images": [
    {
      "image_id": "IMG001",
      "image": "base64_encoded_data",
      "metadata": {...}
    },
    {...}
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "batch_id": "BATCH_001",
  "results": [
    {
      "image_id": "IMG001",
      "classification": {...}
    }
  ],
  "summary": {
    "total": 10,
    "microplastic": 7,
    "biological": 3
  }
}
```

#### **GET /results/{classification_id}**

Retrieve classification by ID

**Response:**
```json
{
  "status": "success",
  "classification": {...}
}
```

#### **GET /statistics**

Get aggregated statistics

**Query Parameters:**
- `start_date`: ISO 8601 date
- `end_date`: ISO 8601 date
- `site_id`: Optional site filter

**Response:**
```json
{
  "status": "success",
  "statistics": {
    "total_particles": 1245,
    "microplastic_percentage": 72.4,
    "polymer_distribution": {...}
  }
}
```

---

## APPENDIX A: Example Data Files

### A.1 Minimal Valid Input (JSON)

```json
{
  "image_metadata": {
    "file_name": "test_particle.png",
    "capture_datetime": "2026-01-31T14:30:52Z"
  },
  "microscope_settings": {
    "scale_um_per_pixel": 0.5
  },
  "environmental_conditions": {
    "location": {
      "latitude": 34.0195,
      "longitude": -118.4912
    }
  }
}
```

### A.2 Complete Example with All Optional Fields

See Section 1.2 for comprehensive example.

---

**Document Version**: 1.0
**Last Updated**: January 31, 2026
**Author**: Micro-Scale Sentinel Team
**Status**: Production Ready