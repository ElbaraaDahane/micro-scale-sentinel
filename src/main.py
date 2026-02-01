import argparse
import sys
import yaml
import os
import cv2
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

from preprocessing import ImagePreprocessor, FeatureExtractor
from classifier import GeminiClassifier
from storage import StorageManager
from reporting import ReportGenerator

# Load env variables
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentinel.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MicroScaleSentinel")


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def process_single_image(image_path: str, config: dict,
                         preprocessor: ImagePreprocessor,
                         extractor: FeatureExtractor,
                         classifier: GeminiClassifier,
                         storage: StorageManager):
    logger.info(f"Processing {image_path}...")

    # 1. Metadata Construction (Simulated for single file)
    image_id = Path(image_path).stem + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    metadata = {
        "image_id": image_id,
        "file_name": os.path.basename(image_path),
        "capture_datetime": datetime.now().isoformat(),
        "scale_um_per_pixel": config['feature_extraction']['scale_um_per_pixel'],
        "location": {"latitude": 0.0, "longitude": 0.0, "depth_m": 5.0},
        "water_properties": {"temperature_c": 20.0, "salinity_ppt": 35.0}
    }

    # 2. Layer 1: Preprocessing & Feature Extraction
    original, processed = preprocessor.process(image_path)
    particles = extractor.extract_all_features(original, processed)

    logger.info(f"Detected {len(particles)} particles in image.")

    results = []

    # 3. Layer 2 & 3: Classification & Storage
    for p in tqdm(particles, desc="Classifying Particles"):
        # Classify
        classification = classifier.classify_particle(p['roi_image'], p['features'])

        # Determine strict classification based on confidence threshold logic
        conf_mp = classification.get('confidence_microplastic', 0)
        thresh_high = config['classification']['confidence_threshold_high']
        thresh_low = config['classification']['confidence_threshold_low']

        # Override classification label if strictly required by config logic,
        # otherwise trust Gemini's "classification" field
        if conf_mp > thresh_high:
            classification['recommendation'] = "DEFINITE"
        elif conf_mp > thresh_low:
            classification['recommendation'] = "PROBABLE"

        # Store
        storage.save_classification(metadata, p, classification)
        results.append(classification)

    return results


def main():
    parser = argparse.ArgumentParser(description="Micro-Scale Sentinel CLI")
    parser.add_argument('--mode', choices=['single', 'batch', 'server'], required=True, help="Operation mode")
    parser.add_argument('--image', help="Path to image file (for single mode)")
    parser.add_argument('--dir', help="Path to directory (for batch mode)")
    parser.add_argument('--config', default='config/config.yaml', help="Path to config file")

    args = parser.parse_args()

    # Initialize System
    config = load_config(args.config)

    preprocessor = ImagePreprocessor(config)
    extractor = FeatureExtractor(config)

    try:
        classifier = GeminiClassifier(config)
    except ValueError as e:
        logger.critical(str(e))
        sys.exit(1)

    storage = StorageManager(config)
    reporter = ReportGenerator(storage)

    # Execution Modes
    if args.mode == 'single':
        if not args.image:
            logger.error("--image argument required for single mode")
            sys.exit(1)

        process_single_image(args.image, config, preprocessor, extractor, classifier, storage)

        # Generate post-run report
        reporter.create_visualizations("data/results")
        reporter.export_pdf_report(f"data/results/report_{Path(args.image).stem}.pdf")

    elif args.mode == 'batch':
        if not args.dir:
            logger.error("--dir argument required for batch mode")
            sys.exit(1)

        image_files = [str(p) for p in Path(args.dir).glob("*") if
                       p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']]

        logger.info(f"Found {len(image_files)} images in batch.")

        for img_path in image_files:
            try:
                process_single_image(img_path, config, preprocessor, extractor, classifier, storage)
            except Exception as e:
                logger.error(f"Failed processing {img_path}: {e}")

        reporter.create_visualizations("data/results")
        reporter.export_pdf_report("data/results/batch_report.pdf")

    elif args.mode == 'server':
        logger.info("Server mode not yet implemented in this version.")


if __name__ == "__main__":
    main()