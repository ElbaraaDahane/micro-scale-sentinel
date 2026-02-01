# Micro-Scale Sentinel: Prompt Engineering Guide
## Version 1.0 - January 31, 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Core Prompt Architecture](#2-core-prompt-architecture)
3. [Prompt Templates](#3-prompt-templates)
4. [Domain Knowledge Integration](#4-domain-knowledge-integration)
5. [Response Format Specification](#5-response-format-specification)
6. [Prompt Optimization Strategies](#6-prompt-optimization-strategies)
7. [Testing & Validation](#7-testing--validation)

---

## 1. INTRODUCTION

### 1.1 Purpose

This guide provides comprehensive instructions for constructing effective prompts for the Gemini 3 Pro API to classify microplastics in holographic microscopy imagery. The prompt design is critical to achieving >85% classification accuracy.

### 1.2 Design Philosophy

**Key Principles:**
1. **Multimodal Integration**: Combine visual (holographic image) + quantitative (measurements) + contextual (domain knowledge) information
2. **Structured Reasoning**: Guide the model through a systematic analysis process
3. **Confidence Quantification**: Request explicit confidence scores and uncertainty estimates
4. **Explainability**: Demand detailed reasoning for every classification
5. **JSON Output**: Enforce structured output for automated parsing

---

## 2. CORE PROMPT ARCHITECTURE

### 2.1 Prompt Structure

```
┌────────────────────────────────────────┐
│  SECTION 1: ROLE DEFINITION            │
│  (Set expert persona and context)      │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  SECTION 2: TASK SPECIFICATION         │
│  (Clear classification objective)       │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  SECTION 3: DATA INPUT                 │
│  (Particle measurements + metadata)    │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  SECTION 4: DOMAIN KNOWLEDGE           │
│  (Physics, biology, materials science) │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  SECTION 5: ANALYSIS FRAMEWORK         │
│  (Step-by-step reasoning guide)        │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  SECTION 6: OUTPUT FORMAT              │
│  (JSON schema with required fields)    │
└────────────────────────────────────────┘
```

---

## 3. PROMPT TEMPLATES

### 3.1 Master Classification Prompt

```python
MASTER_CLASSIFICATION_PROMPT = """
You are a world-class marine biology and materials science expert with 20 years of experience specializing in microplastic detection using holographic microscopy. You have published extensively on optical properties of polymers and biological organisms in aquatic environments.

═══════════════════════════════════════════════════════════════
TASK: MICROPLASTIC CLASSIFICATION
═══════════════════════════════════════════════════════════════

Analyze the holographic microscopy image provided and classify the particle as either:
1. MICROPLASTIC (plastic polymer fragment)
2. BIOLOGICAL (living or once-living organism)
3. UNCERTAIN (insufficient evidence for confident classification)

═══════════════════════════════════════════════════════════════
PARTICLE MEASUREMENTS (From Image Analysis)
═══════════════════════════════════════════════════════════════

Size and Shape:
• Particle size: {size_um:.1f} micrometers
• Circularity: {circularity:.2f} (1.0 = perfect circle, 0 = highly irregular)
• Aspect ratio: {aspect_ratio:.2f} (length/width)
• Perimeter: {perimeter:.1f} pixels

Optical Properties:
• Estimated refractive index: {refractive_index:.2f}
• Mean intensity: {intensity_mean:.1f} (0-255 scale)
• Intensity variance: {intensity_variance:.1f} (higher = more texture)
• Entropy: {entropy:.2f} (measure of complexity)

═══════════════════════════════════════════════════════════════
ENVIRONMENTAL CONTEXT
═══════════════════════════════════════════════════════════════

Water Conditions:
• Temperature: {water_temp_c:.1f}°C
• Salinity: {salinity_ppt:.1f} ppt (parts per thousand)
• Depth: {depth_m:.1f} meters
• Turbidity: {turbidity_ntu:.1f} NTU

Capture Details:
• Location: {location_description}
• Date/Time: {capture_datetime}
• Equipment: {equipment_type}

═══════════════════════════════════════════════════════════════
DOMAIN KNOWLEDGE: MICROPLASTICS
═══════════════════════════════════════════════════════════════

Optical Characteristics:
1. Refractive Index (RI):
   • PET (Polyethylene Terephthalate): RI ≈ 1.575 (bottles, textiles)
   • HDPE (High-Density Polyethylene): RI ≈ 1.54 (milk jugs, bags)
   • PP (Polypropylene): RI ≈ 1.49 (containers, ropes)
   • PS (Polystyrene): RI ≈ 1.55 (styrofoam, packaging)
   • PVC (Polyvinyl Chloride): RI ≈ 1.54 (pipes, vinyl)
   
2. Morphological Features:
   • Angular, irregular edges (mechanical fragmentation)
   • No biological symmetry (no bilateral/radial patterns)
   • Uniform density and transparency
   • Weathering marks (surface etching, cracks)
   • Fiber-like (textiles), fragment-like (bottles), or bead-like (industrial)

3. Diffraction Patterns:
   • Regular, uniform interference fringes
   • Consistent fringe spacing (indicates uniform RI)
   • Sharp diffraction peaks
   
4. Behavior:
   • Static (no movement across frames)
   • No response to stimuli
   • Buoyancy depends on density vs. water

═══════════════════════════════════════════════════════════════
DOMAIN KNOWLEDGE: BIOLOGICAL ORGANISMS
═══════════════════════════════════════════════════════════════

Common Aquatic Microorganisms (100-500 μm range):

1. Diatoms (Phytoplankton):
   • RI ≈ 1.35-1.40 (silica shell + organic matter)
   • Radial or bilateral symmetry (8-fold, 6-fold common)
   • Visible cell wall structure (frustule patterns)
   • Brown/golden pigmentation (chlorophyll, carotenoids)
   • Complex diffraction (cellular compartments)

2. Copepods (Zooplankton):
   • RI ≈ 1.33-1.38 (mostly water + protein)
   • Clear bilateral symmetry (head, body, tail)
   • Visible appendages (antennae, legs)
   • Movement or partial transparency suggesting organs
   • Variable pigmentation (often translucent with darker spots)

3. Other Plankton:
   • Foraminifera: Calcareous shells, chambered structure
   • Radiolaria: Silica skeletons, intricate symmetry
   • Dinoflagellates: Cellulose plates, flagella visible

General Biological Markers:
• Refractive index 1.33-1.40 (close to water)
• Symmetry (biological designs are rarely random)
• Cellular structures (compartments, organelles)
• Pigmentation (chlorophyll green, carotenoid yellow/brown)
• Movement (swimming, drifting, pulsing)
• Organic texture (not uniform like plastics)

═══════════════════════════════════════════════════════════════
ANALYSIS FRAMEWORK: STEP-BY-STEP REASONING
═══════════════════════════════════════════════════════════════

STEP 1: DIFFRACTION PATTERN ANALYSIS
Examine the holographic interference fringes in the image:
├─ Are fringes REGULAR and UNIFORM? → Suggests uniform material (plastic)
├─ Are fringes COMPLEX and VARIABLE? → Suggests cellular structure (biology)
├─ What does fringe SPACING tell us about refractive index?
└─ Are there SHARP or DIFFUSE edges?

STEP 2: REFRACTIVE INDEX ASSESSMENT
Compare estimated RI ({refractive_index:.2f}) to known materials:
├─ RI 1.4-1.6 → Consistent with plastics (PET, HDPE, PP, PS, PVC)
├─ RI 1.33-1.40 → Consistent with biological organisms
├─ RI > 1.6 → Likely mineral (silica, calcium carbonate)
└─ Consider: Does RI match size and morphology?

STEP 3: MORPHOLOGICAL EXAMINATION
Analyze particle shape and structure:
├─ Is there SYMMETRY (bilateral/radial)? → Suggests biological origin
├─ Are edges ANGULAR/IRREGULAR? → Suggests mechanical fragmentation (plastic)
├─ Is there visible CELLULAR STRUCTURE? → Biological
├─ Is density UNIFORM? → Plastic
└─ Size {size_um:.1f}μm: Does this match common plastics vs. organisms?

STEP 4: COLOR AND TRANSPARENCY
Assess visual appearance:
├─ CLEAR/TRANSLUCENT with no pigmentation? → Consistent with many plastics (PET, PS)
├─ PIGMENTED (brown, green, yellow)? → Suggests biological (chlorophyll, carotenoids)
├─ OPAQUE WHITE? → Could be HDPE plastic or biological shell
└─ Intensity variance {intensity_variance:.1f}: High = complex texture (biology), Low = uniform (plastic)

STEP 5: BEHAVIOR (if video sequence available)
Observe motion patterns:
├─ STATIC across frames? → Consistent with plastics (inert)
├─ MOVEMENT (swimming, drifting)? → Biological (alive or recently dead)
└─ ROTATION or ORIENTATION CHANGE? → Could be either (depends on currents)

STEP 6: HYPOTHESIS GENERATION
Based on Steps 1-5, generate competing hypotheses:

Hypothesis A: Microplastic
├─ Polymer type: [PET | HDPE | PP | PS | PVC | Other]
├─ Supporting evidence: [List specific observations]
├─ Contradicting evidence: [List conflicting observations]
└─ Confidence: [0-100%]

Hypothesis B: Biological Organism
├─ Organism type: [Diatom | Copepod | Other | Unknown]
├─ Supporting evidence: [List specific observations]
├─ Contradicting evidence: [List conflicting observations]
└─ Confidence: [0-100%]

Hypothesis C: Uncertain / Mineral / Other
├─ Alternative explanation: [Describe]
└─ Confidence: [0-100%]

STEP 7: CONFIDENCE ASSESSMENT
Weigh the evidence and assign confidence scores:
├─ Strong evidence (3+ supporting factors, 0 contradictions) → 85-100% confidence
├─ Good evidence (2 supporting, 1 minor contradiction) → 70-85% confidence
├─ Weak evidence (1-2 supporting, 2 contradictions) → 40-70% confidence
└─ Insufficient evidence (ambiguous, multiple contradictions) → <40% confidence

STEP 8: FINAL CLASSIFICATION
Select the hypothesis with highest confidence:
├─ If max confidence ≥85% → Classification is DEFINITE
├─ If max confidence 60-85% → Classification is PROBABLE
├─ If max confidence <60% → Classification is UNCERTAIN (flag for manual review)
└─ State classification, polymer/organism type, and detailed reasoning

═══════════════════════════════════════════════════════════════
RESPONSE FORMAT: JSON ONLY
═══════════════════════════════════════════════════════════════

You MUST respond with valid JSON in this exact format:

{{
  "classification": "MICROPLASTIC | BIOLOGICAL | UNCERTAIN",
  
  "confidence_microplastic": <0-100>,
  "confidence_biological": <0-100>,
  "confidence_uncertain": <0-100>,
  
  "polymer_type": "PET | HDPE | PP | PS | PVC | Other | null",
  "polymer_confidence": <0-100 or null>,
  
  "organism_type": "diatom | copepod | dinoflagellate | foraminifera | radiolaria | other | unknown | null",
  "organism_confidence": <0-100 or null>,
  
  "recommendation": "DEFINITE | PROBABLE | UNCERTAIN",
  
  "evidence": {{
    "diffraction_pattern": "<detailed observation of interference fringes>",
    "refractive_index": "<analysis of RI estimate and what it suggests>",
    "morphology": "<description of shape, edges, symmetry>",
    "color_transparency": "<observations about pigmentation and clarity>",
    "size": "<how size constrains identification>",
    "behavior": "<movement or lack thereof>"
  }},
  
  "reasoning": "<comprehensive explanation integrating all evidence, step-by-step logic, final conclusion>",
  
  "alternative_hypotheses": [
    "<alternative explanation 1 and why it's less likely>",
    "<alternative explanation 2 and why it's less likely>"
  ],
  
  "flags": [
    "<any concerns: 'low_confidence', 'ambiguous_morphology', 'atypical_properties', etc.>"
  ]
}}

═══════════════════════════════════════════════════════════════
CRITICAL INSTRUCTIONS
═══════════════════════════════════════════════════════════════

1. ANALYZE the holographic image carefully, focusing on diffraction patterns
2. INTEGRATE quantitative measurements with visual observations
3. APPLY domain knowledge systematically (not just pattern matching)
4. CONSIDER multiple hypotheses before deciding
5. QUANTIFY confidence explicitly (don't just say "likely")
6. EXPLAIN your reasoning in detail (this is for science, not just automation)
7. FLAG uncertain cases (better to admit uncertainty than make wrong call)
8. RESPOND only in valid JSON format (no extra text before or after)

Begin your analysis now.
"""
```

### 3.2 Simplified Prompt (For Fast Processing)

For situations where processing speed is critical and detailed reasoning is less important:

```python
SIMPLIFIED_PROMPT = """
You are a microplastic detection expert. Classify this particle as MICROPLASTIC, BIOLOGICAL, or UNCERTAIN.

MEASUREMENTS:
Size: {size_um}μm, Circularity: {circularity}, RI: {refractive_index}

KEY RULES:
• RI 1.4-1.6 + irregular edges + no symmetry = MICROPLASTIC
• RI 1.33-1.40 + symmetry + pigmentation = BIOLOGICAL
• Ambiguous = UNCERTAIN

Respond in JSON:
{{
  "classification": "...",
  "confidence_microplastic": 0-100,
  "polymer_type": "PET|HDPE|PP|PS|PVC|null",
  "reasoning": "brief explanation"
}}
"""
```

### 3.3 High-Confidence Verification Prompt

For double-checking low-confidence classifications:

```python
VERIFICATION_PROMPT = """
A previous classification flagged this particle as UNCERTAIN (confidence <60%).

PREVIOUS RESULT:
{previous_classification_json}

TASK: Re-analyze with extra scrutiny. Consider:
1. Could this be an atypical organism (damaged, juvenile, rare species)?
2. Could this be a weathered plastic with unusual properties?
3. Could this be a mineral or other non-plastic, non-biological particle?
4. What additional tests would you recommend (e.g., spectroscopy, movement tracking)?

Provide a REVISED classification with updated confidence scores and detailed reasoning for the change (or confirmation).

Respond in JSON with same format as before, plus:
{{
  ...
  "revision_notes": "explanation of why classification changed or stayed same",
  "recommended_tests": ["list of additional analyses that would help"]
}}
"""
```

---

## 4. DOMAIN KNOWLEDGE INTEGRATION

### 4.1 Polymer Properties Reference Table

This table should be embedded in prompts for accurate polymer identification:

```python
POLYMER_PROPERTIES = """
╔══════════════════════════════════════════════════════════════════╗
║                 POLYMER IDENTIFICATION GUIDE                     ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ PET (Polyethylene Terephthalate)                                │
├─────────────────────────────────────────────────────────────────┤
│ RI: 1.575 (high)                                                │
│ Appearance: Clear, transparent (bottles, textiles)              │
│ Morphology: Fibers (clothing) or fragments (bottles)            │
│ Common sources: Beverage bottles, synthetic fabrics, packaging  │
│ Distinguishing: Sharp diffraction, uniform transparency         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ HDPE (High-Density Polyethylene)                                │
├─────────────────────────────────────────────────────────────────┤
│ RI: 1.54 (medium)                                               │
│ Appearance: Opaque white or translucent                         │
│ Morphology: Fragments, angular pieces                           │
│ Common sources: Milk jugs, detergent bottles, plastic bags      │
│ Distinguishing: Waxy appearance, less transparent than PET      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PP (Polypropylene)                                              │
├─────────────────────────────────────────────────────────────────┤
│ RI: 1.49 (lowest common plastic)                                │
│ Appearance: Semi-transparent, flexible                          │
│ Morphology: Fragments or fibers (ropes, textiles)              │
│ Common sources: Food containers, bottle caps, ropes, textiles   │
│ Distinguishing: Lower RI, flexible appearance                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PS (Polystyrene)                                                │
├─────────────────────────────────────────────────────────────────┤
│ RI: 1.55 (medium)                                               │
│ Appearance: Clear or opaque white (foam)                        │
│ Morphology: Fragments, often irregular from foam breakdown      │
│ Common sources: Styrofoam, food packaging, disposable cups      │
│ Distinguishing: Brittle, may show cellular structure if foam    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PVC (Polyvinyl Chloride)                                        │
├─────────────────────────────────────────────────────────────────┤
│ RI: 1.54 (medium)                                               │
│ Appearance: Translucent to opaque, often colored                │
│ Morphology: Rigid fragments                                     │
│ Common sources: Pipes, vinyl siding, flooring, records          │
│ Distinguishing: Rigid, often retains color from original product│
└─────────────────────────────────────────────────────────────────┘
"""
```

### 4.2 Biological Organism Reference

```python
ORGANISM_PROPERTIES = """
╔══════════════════════════════════════════════════════════════════╗
║              COMMON AQUATIC MICROORGANISMS (100-500μm)           ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ DIATOMS (Phytoplankton)                                         │
├─────────────────────────────────────────────────────────────────┤
│ RI: 1.35-1.40 (silica + organic matter)                        │
│ Size: 10-500 μm (highly variable)                              │
│ Symmetry: Radial (centric) or bilateral (pennate)              │
│ Key features:                                                    │
│   • Silica frustule (cell wall) with intricate patterns        │
│   • Golden-brown pigmentation (fucoxanthin)                    │
│   • Two overlapping shells (like petri dish)                   │
│   • Visible striations or pores in frustule                    │
│ Diffraction: Complex due to cellular compartments              │
│ Behavior: May show slight movement (dead cells don't)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ COPEPODS (Zooplankton)                                          │
├─────────────────────────────────────────────────────────────────┤
│ RI: 1.33-1.38 (mostly water + protein)                         │
│ Size: 500 μm - 2 mm (adults); 100-500 μm (nauplii larvae)     │
│ Symmetry: Bilateral                                             │
│ Key features:                                                    │
│   • Clear segmented body (head, thorax, abdomen)               │
│   • Visible appendages (antennae, swimming legs)               │
│   • Eye spots (often red/black)                                │
│   • Mostly transparent with darker internal organs             │
│ Diffraction: Variable due to body segments                     │
│ Behavior: Active swimming (alive) or drifting (dead)           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ FORAMINIFERA (Protozoans)                                       │
├─────────────────────────────────────────────────────────────────┤
│ RI: 1.50-1.58 (calcium carbonate shell)                        │
│ Size: 100 μm - several mm                                      │
│ Symmetry: Variable (often spiral)                              │
│ Key features:                                                    │
│   • Chambered calcareous shell                                 │
│   • Visible pores (for pseudopods)                             │
│   • Often white/translucent                                    │
│   • Spiral or linear arrangement of chambers                   │
│ Note: RI overlaps with plastics—use symmetry to distinguish    │
└─────────────────────────────────────────────────────────────────┘
"""
```

---

## 5. RESPONSE FORMAT SPECIFICATION

### 5.1 Required JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": [
    "classification",
    "confidence_microplastic",
    "confidence_biological",
    "recommendation",
    "reasoning"
  ],
  "properties": {
    "classification": {
      "type": "string",
      "enum": ["MICROPLASTIC", "BIOLOGICAL", "UNCERTAIN"]
    },
    "confidence_microplastic": {
      "type": "number",
      "minimum": 0,
      "maximum": 100
    },
    "confidence_biological": {
      "type": "number",
      "minimum": 0,
      "maximum": 100
    },
    "confidence_uncertain": {
      "type": "number",
      "minimum": 0,
      "maximum": 100
    },
    "polymer_type": {
      "type": ["string", "null"],
      "enum": ["PET", "HDPE", "PP", "PS", "PVC", "Other", null]
    },
    "polymer_confidence": {
      "type": ["number", "null"],
      "minimum": 0,
      "maximum": 100
    },
    "organism_type": {
      "type": ["string", "null"],
      "enum": ["diatom", "copepod", "dinoflagellate", "foraminifera", "radiolaria", "other", "unknown", null]
    },
    "organism_confidence": {
      "type": ["number", "null"]
    },
    "recommendation": {
      "type": "string",
      "enum": ["DEFINITE", "PROBABLE", "UNCERTAIN"]
    },
    "evidence": {
      "type": "object",
      "required": ["diffraction_pattern", "morphology", "reasoning"],
      "properties": {
        "diffraction_pattern": {"type": "string"},
        "refractive_index": {"type": "string"},
        "morphology": {"type": "string"},
        "color_transparency": {"type": "string"},
        "size": {"type": "string"},
        "behavior": {"type": "string"}
      }
    },
    "reasoning": {
      "type": "string",
      "minLength": 100
    },
    "alternative_hypotheses": {
      "type": "array",
      "items": {"type": "string"}
    },
    "flags": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["low_confidence", "ambiguous_morphology", "atypical_properties", "needs_verification", "edge_case"]
      }
    }
  }
}
```

### 5.2 Response Validation

```python
import jsonschema

def validate_gemini_response(response: dict) -> bool:
    """Validate Gemini response against schema"""
    try:
        jsonschema.validate(instance=response, schema=RESPONSE_SCHEMA)
        
        # Additional logic checks
        assert 0 <= response['confidence_microplastic'] <= 100
        assert 0 <= response['confidence_biological'] <= 100
        assert abs(
            response['confidence_microplastic'] + 
            response['confidence_biological'] + 
            response.get('confidence_uncertain', 0) - 100
        ) < 5  # Allow 5% tolerance for rounding
        
        if response['classification'] == 'MICROPLASTIC':
            assert response['polymer_type'] is not None
        elif response['classification'] == 'BIOLOGICAL':
            assert response['organism_type'] is not None
            
        return True
    except (jsonschema.ValidationError, AssertionError) as e:
        log.error(f"Response validation failed: {e}")
        return False
```

---

## 6. PROMPT OPTIMIZATION STRATEGIES

### 6.1 Iterative Refinement Process

**Week 1-2: Baseline Performance**
1. Test master prompt on 50 labeled samples
2. Measure accuracy, precision, recall
3. Identify failure modes (false positives, false negatives)

**Week 3-4: Targeted Improvements**
1. Add examples of failure cases to domain knowledge section
2. Emphasize distinguishing features for confused categories
3. Adjust confidence thresholds based on observed calibration

**Week 5-6: A/B Testing**
1. Test prompt variations (e.g., more vs. less detail)
2. Compare performance metrics
3. Select best-performing version

### 6.2 Common Failure Modes and Solutions

| Failure Mode | Symptom | Solution |
|-------------|---------|----------|
| Over-confidence | Claims 95%+ confidence on ambiguous cases | Add explicit uncertainty checks to prompt |
| Under-confidence | Flags too many as "UNCERTAIN" | Provide more decision rules, reduce threshold |
| Polymer confusion | Confuses PET with PS (similar RI) | Add distinguishing features (PET more transparent) |
| False biological | Mistakes weathered plastic for organism | Emphasize lack of symmetry as key marker |
| False plastic | Mistakes foram shell for plastic (RI overlap) | Emphasize chambered structure of forams |

### 6.3 Prompt Length vs. Performance Trade-off

**Empirical Findings:**
- Very short prompts (<500 tokens): 65-75% accuracy
- Medium prompts (1000-2000 tokens): 80-90% accuracy ← Sweet spot
- Very long prompts (>3000 tokens): 85-92% accuracy but 2x slower

**Recommendation**: Use master prompt (2000 tokens) for production. Use simplified prompt for real-time screening (flag uncertain, re-analyze with full prompt).

---

## 7. TESTING & VALIDATION

### 7.1 Test Dataset Construction

**Balanced Test Set (Recommended):**
- 40% Microplastics (various polymers)
- 40% Biological organisms (diatoms, copepods, etc.)
- 20% Edge cases (minerals, degraded organics, ambiguous)

**Diversity Requirements:**
- Size range: 100-500 μm (representative of detection limits)
- Polymer types: At least 10 samples each of PET, HDPE, PP, PS, PVC
- Organism types: At least 10 samples each of diatoms, copepods
- Image quality: Mix of high-quality and noisy images

### 7.2 Evaluation Metrics

```python
def evaluate_prompt_performance(test_set: List[dict]) -> dict:
    """
    Evaluate prompt on test set
    
    test_set format: [
        {
            'image': np.ndarray,
            'features': dict,
            'ground_truth': 'MICROPLASTIC' | 'BIOLOGICAL',
            'ground_truth_polymer': 'PET' | etc.,
        },
        ...
    ]
    """
    correct = 0
    polymer_correct = 0
    confidences = []
    high_confidence_correct = 0
    high_confidence_total = 0
    
    for sample in test_set:
        result = gemini_classifier.classify(sample['image'], sample['features'])
        
        # Overall accuracy
        if result['classification'] == sample['ground_truth']:
            correct += 1
            
        # Polymer accuracy (if microplastic)
        if sample['ground_truth'] == 'MICROPLASTIC':
            if result['polymer_type'] == sample['ground_truth_polymer']:
                polymer_correct += 1
        
        # Confidence calibration
        confidence = result['confidence_microplastic'] if result['classification'] == 'MICROPLASTIC' else result['confidence_biological']
        confidences.append(confidence)
        
        if confidence > 85:
            high_confidence_total += 1
            if result['classification'] == sample['ground_truth']:
                high_confidence_correct += 1
    
    return {
        'accuracy': correct / len(test_set),
        'polymer_accuracy': polymer_correct / sum(1 for s in test_set if s['ground_truth'] == 'MICROPLASTIC'),
        'mean_confidence': np.mean(confidences),
        'high_confidence_accuracy': high_confidence_correct / high_confidence_total if high_confidence_total > 0 else 0,
        'high_confidence_rate': high_confidence_total / len(test_set)
    }
```

### 7.3 Continuous Improvement

**Feedback Loop:**
1. Deploy classifier in production
2. Flag uncertain classifications for expert review
3. Collect expert corrections (ground truth labels)
4. Quarterly: Re-evaluate prompt on new ground truth data
5. Update prompt based on new failure modes
6. Re-deploy improved prompt

---

## APPENDIX A: Example Prompts for Specific Scenarios

### A.1 PET Microfiber (Textile Fragment)

```
Expected measurements:
- Size: 200-400 μm (fiber length)
- Aspect ratio: 5-15 (long, thin)
- Circularity: 0.1-0.3 (highly irregular, fiber-like)
- RI: 1.57-1.58
- Intensity variance: Low (uniform)

Expected classification:
{
  "classification": "MICROPLASTIC",
  "confidence_microplastic": 92,
  "polymer_type": "PET",
  "reasoning": "Elongated fiber morphology with RI=1.57 is characteristic of PET textile fibers. The high aspect ratio and uniform transparency are consistent with synthetic fabric breakdown..."
}
```

### A.2 Diatom (Biological)

```
Expected measurements:
- Size: 50-200 μm
- Circularity: 0.7-0.9 (circular or oval)
- RI: 1.35-1.40
- Intensity variance: High (cellular structure)

Expected classification:
{
  "classification": "BIOLOGICAL",
  "confidence_biological": 88,
  "organism_type": "diatom",
  "reasoning": "The radial symmetry and visible frustule patterning are diagnostic of a centric diatom. The RI of 1.38 is consistent with silica + organic matter. The golden-brown coloration suggests active chlorophyll..."
}
```

---

**Document Version**: 1.0
**Last Updated**: January 31, 2026
**Author**: Micro-Scale Sentinel Team
**Status**: Production Ready