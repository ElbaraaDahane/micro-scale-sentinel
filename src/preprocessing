import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Layer 1: Handles low-level image enhancement to prepare holographic
    microscopy data for feature extraction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing preprocessing parameters.
        """
        self.config = config
        self.clahe_clip = config['preprocessing']['clahe']['clip_limit']
        self.grid_size = tuple(config['preprocessing']['clahe']['tile_grid_size'])
        self.bilateral_d = config['preprocessing']['bilateral']['d']
        self.sigma_color = config['preprocessing']['bilateral']['sigma_color']
        self.sigma_space = config['preprocessing']['bilateral']['sigma_space']

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.grid_size)
            enhanced = clahe.apply(gray)
            return enhanced
        except Exception as e:
            logger.error(f"Error in enhance_contrast: {e}")
            raise

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Applies Bilateral Filter to preserve edges while reducing noise."""
        try:
            return cv2.bilateralFilter(
                image,
                self.bilateral_d,
                self.sigma_color,
                self.sigma_space
            )
        except Exception as e:
            logger.error(f"Error in reduce_noise: {e}")
            raise

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalizes pixel values to [0, 255] range."""
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    def process(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.

        Returns:
            Tuple[original_color_image, processed_gray_image]
        """
        try:
            original = cv2.imread(image_path)
            if original is None:
                raise ValueError(f"Could not load image at {image_path}")

            # enhancement pipeline
            enhanced = self.enhance_contrast(original)
            denoised = self.reduce_noise(enhanced)
            normalized = self.normalize(denoised)

            return original, normalized
        except Exception as e:
            logger.error(f"Pipeline failed for {image_path}: {e}")
            raise


class FeatureExtractor:
    """
    Extracts physics-based geometric and optical features from preprocessed images.
    """

    def __init__(self, config: Dict[str, Any]):
        self.scale_um = config['feature_extraction']['scale_um_per_pixel']
        self.min_size = config['feature_extraction']['min_particle_size_um']
        self.max_size = config['feature_extraction']['max_particle_size_um']

    def estimate_refractive_index(self, roi: np.ndarray) -> float:
        """
        Heuristic estimation of Refractive Index (RI) based on edge intensity and contrast.
        In holographic microscopy, higher RI differences often cause stronger phase
        shifts resulting in sharper, higher contrast diffraction fringes.

        Returns:
            float: Estimated RI proxy (unitless, calibrated to range 1.3-1.7)
        """
        try:
            # Calculate gradient magnitude (edge strength)
            sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

            mean_gradient = np.mean(gradient_magnitude)
            contrast = np.max(roi) - np.min(roi)

            # Linear mapping heuristic (calibrated to approximate physical RI ranges)
            # This is a simplification for the CV layer; Gemini does the heavy reasoning.
            # Assume base water RI ~1.33. Add offset based on edge hardness.
            ri_estimate = 1.33 + (mean_gradient / 255.0) * 0.4 + (contrast / 255.0) * 0.1

            return float(np.clip(ri_estimate, 1.33, 1.75))
        except Exception:
            return 1.33

    def extract_texture_features(self, roi: np.ndarray) -> Dict[str, float]:
        """Calculates texture metrics like entropy and variance."""
        try:
            # Shannon Entropy
            marg = np.histogramdd(np.ravel(roi), bins=256)[0] / roi.size
            marg = list(filter(lambda p: p > 0, np.ravel(marg)))
            entropy = -np.sum(np.array(marg) * np.log2(np.array(marg)))

            return {
                "intensity_mean": float(np.mean(roi)),
                "intensity_variance": float(np.var(roi)),
                "entropy": float(entropy),
                "contrast": float(np.max(roi) - np.min(roi))
            }
        except Exception:
            return {"intensity_mean": 0.0, "intensity_variance": 0.0, "entropy": 0.0, "contrast": 0.0}

    def extract_all_features(self, original: np.ndarray, processed: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects particles and extracts features for each.

        Returns:
            List of dictionaries containing feature vectors and ROI data.
        """
        particles = []

        # Thresholding to find objects
        _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            area_pixels = cv2.contourArea(cnt)
            # Convert to physical size (assuming approx circular for diameter estimate)
            # Area in um^2 = pixels * scale^2
            area_um2 = area_pixels * (self.scale_um ** 2)
            equivalent_diameter_um = 2 * np.sqrt(area_um2 / np.pi)

            if not (self.min_size <= equivalent_diameter_um <= self.max_size):
                continue

            # Geometry
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area_pixels / (perimeter ** 2)

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h != 0 else 0

            # ROI Extraction (with padding)
            pad = 10
            h_img, w_img = processed.shape
            y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
            x1, x2 = max(0, x - pad), min(w_img, x + w + pad)

            roi_gray = processed[y1:y2, x1:x2]
            roi_color = original[y1:y2, x1:x2]

            if roi_gray.size == 0: continue

            # Optical Features
            ri_est = self.estimate_refractive_index(roi_gray)
            texture = self.extract_texture_features(roi_gray)

            particle_data = {
                "particle_id": f"P{i:04d}",
                "bbox": (x, y, w, h),
                "roi_image": roi_color,  # For passing to Gemini
                "features": {
                    "size_um": float(equivalent_diameter_um),
                    "area_um2": float(area_um2),
                    "circularity": float(circularity),
                    "aspect_ratio": float(aspect_ratio),
                    "refractive_index_estimate": ri_est,
                    **texture
                }
            }
            particles.append(particle_data)

        logger.info(f"Extracted features for {len(particles)} particles.")
        return particles


if __name__ == "__main__":
    # Test block
    dummy_conf = {
        'preprocessing': {'clahe': {'clip_limit': 2.0, 'tile_grid_size': [8, 8]},
                          'bilateral': {'d': 9, 'sigma_color': 75, 'sigma_space': 75}},
        'feature_extraction': {'scale_um_per_pixel': 0.5, 'min_particle_size_um': 10, 'max_particle_size_um': 1000}
    }
    # Create a dummy image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(img, (250, 250), 50, (200, 200, 200), -1)

    pre = ImagePreprocessor(dummy_conf)
    feat = FeatureExtractor(dummy_conf)

    enh = pre.enhance_contrast(img)
    den = pre.reduce_noise(enh)
    nrm = pre.normalize(den)

    data = feat.extract_all_features(img, nrm)
    print(f"Found {len(data)} particles")
