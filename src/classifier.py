import os
import json
import logging
import time
import google.generativeai as genai
from typing import Dict, Any, Optional
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiClassifier:
    """
    Layer 2: Interfaces with Google Gemini API to perform multimodal classification
    of microparticles based on visual data and extracted feature vectors.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        genai.configure(api_key=self.api_key)
        self.model_name = config['gemini']['model']
        self.model = genai.GenerativeModel(self.model_name)

        self.max_retries = config['gemini']['max_retries']
        self.timeout = config['gemini']['timeout_sec']

    def build_prompt(self, features: Dict[str, Any]) -> str:
        """
        Constructs the expert prompt containing role definition, domain knowledge,
        and specific feature data for the particle being analyzed.
        """
        feature_json = json.dumps(features, indent=2)

        prompt = f"""
**SECTION 1: ROLE DEFINITION**
You are a world-class marine biology and materials science expert with 20 years of experience in microplastic detection using holographic microscopy. Your specialty lies in distinguishing synthetic anthropogenic polymers from natural biological matter in aquatic samples. You analyze images and numerical feature vectors to make high-stakes classification decisions.

**SECTION 2: DOMAIN KNOWLEDGE - POLYMERS (MICROPLASTICS)**
You must apply your knowledge of the following common pollutants:
1. **PET (Polyethylene Terephthalate)**: 
   - Refractive Index (RI): ~1.575
   - Morphology: Often fibers from textiles or irregular fragments from bottles. 
   - Appearance: Clear, uniform texture, high contrast edges due to high RI difference with water.
2. **HDPE (High-Density Polyethylene)**:
   - RI: ~1.54
   - Morphology: Irregular fragments.
   - Appearance: Often opaque or semi-opaque white/colored.
3. **PP (Polypropylene)**:
   - RI: ~1.49
   - Morphology: Films, fragments, or fibers.
   - Appearance: Translucent, waxy surface texture.
4. **PS (Polystyrene)**:
   - RI: ~1.55-1.59
   - Morphology: Brittle fragments with sharp, conchoidal fractures.
5. **PVC (Polyvinyl Chloride)**:
   - RI: ~1.54
   - Morphology: Rigid fragments.
   - Appearance: Often colored, dense.

**SECTION 3: DOMAIN KNOWLEDGE - BIOLOGY**
You must distinguish the above from common look-alikes:
1. **Diatoms**:
   - RI: 1.35-1.42 (Silica frustules ~1.46 but effective RI lower due to water content)
   - Morphology: Geometric perfection, radial or bilateral symmetry, pores/striae visible.
   - Appearance: Often contain golden/brown pigments (chloroplasts).
2. **Copepods/Zooplankton**:
   - RI: 1.33-1.38 (Close to water)
   - Morphology: Distinct body segments, appendages, antennae.
   - Appearance: High transparency, soft edges, internal organs visible.

**SECTION 4: ANALYSIS FRAMEWORK**
Perform the analysis in these strict steps:
1. **Diffraction & RI Analysis**: Look at the image fringe patterns. High contrast, sharp fringes imply high RI (Plastic). Soft, halo-like fringes imply low RI (Biological). Compare the calculated `refractive_index_estimate` in the input data to the known values above.
2. **Morphological Screening**: Look for symmetry. Biology evolves symmetry; plastics break randomly (irregular, sharp corners) or are extruded (perfect fibers).
3. **Internal Structure**: Does the object have internal organs/chloroplasts (Bio) or is it homogeneous/uniform (Plastic)?
4. **Data Synthesis**: Combine the visual evidence with the provided feature vector: {feature_json}.
5. **Confidence Scoring**: Assign 0-100 scores based on evidence strength.

**SECTION 5: OUTPUT SCHEMA**
You must return ONLY a valid JSON object. Do not include markdown code blocks. The JSON must match this structure:
{{
  "classification": "MICROPLASTIC" or "BIOLOGICAL" or "UNCERTAIN",
  "confidence_microplastic": <0-100>,
  "confidence_biological": <0-100>,
  "polymer_type": "PET" | "HDPE" | "PP" | "PS" | "PVC" | "Other" | null,
  "organism_type": "diatom" | "copepod" | "other" | null,
  "recommendation": "DEFINITE" | "PROBABLE" | "UNCERTAIN",
  "reasoning": "A detailed 2-3 sentence explanation citing specific observed features.",
  "evidence": {{
    "diffraction_pattern": "Describe intensity and sharpness",
    "refractive_index_analysis": "Assessment of RI match",
    "morphology": "Shape description",
    "size_analysis": "Comment on size consistency"
  }}
}}
"""
        return prompt

    def classify_particle(self, particle_image: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the cropped particle image and feature data to Gemini for classification.
        """
        # Convert CV2 BGR to RGB then to PIL Image
        rgb_image = cv2.cvtColor(particle_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        prompt = self.build_prompt(features)

        for attempt in range(self.max_retries):
            try:
                # Use generation_config to enforce JSON response (available in newer models)
                response = self.model.generate_content(
                    [prompt, pil_image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,  # Low temperature for analytical precision
                        response_mime_type="application/json"
                    )
                )

                return self.parse_response(response.text)

            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                time.sleep(2 * (attempt + 1))  # Exponential backoff

        # Fallback if all retries fail
        logger.error("All Gemini API retries failed.")
        return self._get_fallback_response()

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parses and validates the JSON response from Gemini."""
        try:
            # Clean up potential markdown formatting if model ignores mime_type
            clean_text = response_text.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_text)

            # Basic validation
            required_keys = ["classification", "confidence_microplastic", "polymer_type", "reasoning"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing key in response: {key}")

            return data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {response_text}")
            return self._get_fallback_response()
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return self._get_fallback_response()

    def _get_fallback_response(self) -> Dict[str, Any]:
        return {
            "classification": "UNCERTAIN",
            "confidence_microplastic": 0,
            "confidence_biological": 0,
            "polymer_type": None,
            "organism_type": None,
            "recommendation": "UNCERTAIN",
            "reasoning": "API Classification failed.",
            "evidence": {}
        }


if __name__ == "__main__":
    # Mock usage
    pass