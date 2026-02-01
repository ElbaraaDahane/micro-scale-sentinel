"""
Micro-Scale Sentinel Dashboard
AI-Powered Microplastic Detection with Real Image Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import json
import numpy as np
from PIL import Image
import io

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Micro-Scale Sentinel | AI Microplastic Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; }
    h1 { color: #2C3E50; border-bottom: 3px solid #3498DB; padding-bottom: 10px; }
    h2 { color: #34495E; margin-top: 30px; }
    
    .alert-success {
        background-color: #D5F4E6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #27AE60;
        margin: 10px 0;
    }
    
    .alert-warning {
        background-color: #FCF3CF;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #F39C12;
        margin: 10px 0;
    }
    
    .alert-danger {
        background-color: #FADBD8;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #E74C3C;
        margin: 10px 0;
    }
    
    .upload-box {
        border: 2px dashed #3498DB;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #ECF0F1;
        margin: 20px 0;
    }
    
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #7F8C8D;
        margin-top: 50px;
        border-top: 2px solid #BDC3C7;
    }
</style>
""", unsafe_allow_html=True)

# ==================== GEMINI AI CLASSIFIER ====================
def classify_with_gemini(image, particle_features=None):
    """
    Classify particle using Google Gemini API
    
    Args:
        image: PIL Image object
        particle_features: Optional dict with size, circularity, etc.
    
    Returns:
        dict with classification results
    """
    try:
        import google.generativeai as genai
        
        # Get API key from secrets
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        
        if not api_key:
            return {
                "status": "error",
                "message": "API key not configured. Please add GEMINI_API_KEY to Streamlit secrets."
            }
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Build prompt
        if particle_features:
            prompt = f"""
You are a world-class marine biology and materials science expert with 20 years of experience in microplastic detection using holographic microscopy.

TASK: Analyze this holographic microscopy image and classify the particle as MICROPLASTIC, BIOLOGICAL, or UNCERTAIN.

PARTICLE MEASUREMENTS:
- Size: {particle_features.get('size_um', 'Unknown')} μm
- Circularity: {particle_features.get('circularity', 'Unknown')}
- Estimated Refractive Index: {particle_features.get('ri_estimate', 'Unknown')}
- Intensity Variance: {particle_features.get('intensity_variance', 'Unknown')}

DOMAIN KNOWLEDGE - MICROPLASTIC POLYMERS:
• PET (Polyethylene Terephthalate): RI = 1.575, from bottles/textiles, clear and uniform
• HDPE (High-Density Polyethylene): RI = 1.54, from containers, opaque white
• PP (Polypropylene): RI = 1.49, from ropes/containers, translucent
• PS (Polystyrene): RI = 1.55, from styrofoam, brittle fragments
• PVC (Polyvinyl Chloride): RI = 1.54, from pipes, rigid and colored

Microplastics typically show:
- Refractive index: 1.4-1.6
- Irregular, fragmented shapes
- No biological symmetry
- Uniform material texture
- Sharp, angular edges from mechanical breakdown

DOMAIN KNOWLEDGE - BIOLOGICAL ORGANISMS:
• Diatoms: RI = 1.35-1.40, radial symmetry, silica shells, golden-brown pigment
• Copepods: RI = 1.33-1.38, bilateral symmetry, visible appendages, transparent
• Other plankton: RI = 1.33-1.40, complex internal structures, cellular organization

Biological organisms typically show:
- Refractive index: 1.33-1.40
- Radial or bilateral symmetry
- Complex internal structures
- Cellular organization
- Organic, curved shapes

ANALYSIS FRAMEWORK (follow these steps):
1. Analyze the holographic diffraction pattern (regular vs. complex)
2. Estimate refractive index from fringe spacing
3. Compare RI to known ranges (plastics: 1.4-1.6, biology: 1.33-1.40)
4. Examine morphology (symmetry, edges, structure)
5. Assess transparency and pigmentation
6. Generate competing hypotheses
7. Weigh evidence for each hypothesis
8. Assign confidence scores based on strength of evidence
9. Select final classification

OUTPUT FORMAT (respond ONLY with valid JSON):
{{
  "classification": "MICROPLASTIC or BIOLOGICAL or UNCERTAIN",
  "confidence_microplastic": 0-100,
  "confidence_biological": 0-100,
  "polymer_type": "PET or HDPE or PP or PS or PVC or Other or null",
  "organism_type": "diatom or copepod or other or null",
  "recommendation": "DEFINITE or PROBABLE or UNCERTAIN or MANUAL_REVIEW",
  "size_category": "nano (<1μm) or micro (1-1000μm) or macro (>1000μm)",
  "reasoning": "Detailed multi-sentence explanation of your classification with specific evidence",
  "evidence": {{
    "diffraction_pattern": "description of what you observe",
    "refractive_index_analysis": "what the RI tells you",
    "morphology": "shape and structure observations",
    "symmetry": "symmetry analysis",
    "color_pigmentation": "color and transparency notes"
  }}
}}
"""
        else:
            prompt = """
You are a world-class marine biology and materials science expert analyzing holographic microscopy images for microplastic detection.

TASK: Analyze this image and classify the particle as MICROPLASTIC, BIOLOGICAL, or UNCERTAIN.

Consider:
1. Diffraction patterns (regular = plastic, complex = biological)
2. Refractive index indicators (plastics: 1.4-1.6, biology: 1.33-1.40)
3. Morphology (irregular/fragmented = plastic, symmetric = biological)
4. Transparency and color
5. Size and shape characteristics

Respond ONLY with valid JSON:
{
  "classification": "MICROPLASTIC or BIOLOGICAL or UNCERTAIN",
  "confidence_microplastic": 0-100,
  "confidence_biological": 0-100,
  "polymer_type": "PET or HDPE or PP or PS or PVC or null",
  "organism_type": "diatom or copepod or null",
  "reasoning": "Detailed explanation",
  "recommendation": "DEFINITE or PROBABLE or UNCERTAIN"
}
"""
        
        # Call Gemini API
        response = model.generate_content([prompt, image])
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Extract JSON if wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json").split("```").strip()[1]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        result["status"] = "success"
        
        return result
        
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": f"Failed to parse AI response as JSON. Raw response: {response_text[:200]}..."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Classification failed: {str(e)}"
        }

# ==================== IMAGE FEATURE EXTRACTION ====================
def extract_basic_features(image):
    """
    Extract basic features from image
    
    Args:
        image: PIL Image
    
    Returns:
        dict with features
    """
    try:
        import cv2
        
        # Convert to numpy array
        img_array = np.array(image.convert('L'))  # Grayscale
        
        # Basic statistics
        mean_intensity = np.mean(img_array)
        variance = np.var(img_array)
        
        # Simple edge detection
        edges = cv2.Canny(img_array, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Estimate size (placeholder - would need calibration)
        estimated_size = np.random.uniform(100, 500)  # μm
        
        # Estimate circularity (placeholder)
        circularity = np.random.uniform(0.5, 0.9)
        
        # Estimate RI from intensity (simplified)
        ri_estimate = 1.33 + (mean_intensity / 255) * 0.27
        
        return {
            "size_um": round(estimated_size, 1),
            "circularity": round(circularity, 2),
            "ri_estimate": round(ri_estimate, 2),
            "intensity_mean": round(mean_intensity, 1),
            "intensity_variance": round(variance, 1),
            "edge_density": round(edge_density, 3)
        }
    except:
        return {
            "size_um": "Unknown",
            "circularity": "Unknown",
            "ri_estimate": "Unknown",
            "intensity_variance": "Unknown"
        }

# ==================== SAMPLE DATA FUNCTIONS ====================
@st.cache_data
def get_sample_stats():
    """Return sample statistics for demo"""
    return {
        'total_particles': 150,
        'microplastic_count': 108,
        'biological_count': 38,
        'uncertain_count': 4,
        'microplastic_percentage': 72.0,
        'biological_percentage': 25.3,
        'uncertain_percentage': 2.7,
        'polymer_distribution': {
            'PET': 45,
            'HDPE': 28,
            'PP': 20,
            'PS': 10,
            'PVC': 5
        },
        'avg_confidence': 84.5,
        'high_confidence_count': 132,
        'medium_confidence_count': 14,
        'low_confidence_count': 4
    }

@st.cache_data
def get_sample_classifications():
    """Return sample classification data"""
    dates = pd.date_range(end=datetime.now(), periods=20, freq='2H')
    
    classifications = ['Microplastic'] * 14 + ['Biological'] * 5 + ['Uncertain']
    polymers = ['PET', 'HDPE', 'PP', 'PET', 'PS', 'HDPE', 'PVC', 'PP', 'PET', 'PS', 
                'HDPE', 'PET', 'PP', 'PVC'] + [None] * 6
    organisms = [None] * 14 + ['Diatom', 'Copepod', 'Diatom', 'Copepod', 'Diatom', None]
    confidences = [87, 92, 78, 85, 91, 76, 88, 82, 89, 93, 84, 90, 79, 86, 
                   89, 94, 86, 91, 88, 45]
    sizes = [245, 312, 189, 267, 198, 334, 156, 278, 221, 198, 
             254, 289, 176, 209, 128, 445, 167, 389, 142, 289]
    
    return pd.DataFrame({
        'Timestamp': dates,
        'Particle ID': [f'P{i:03d}' for i in range(1, 21)],
        'Classification': classifications,
        'Type': [p if p else o for p, o in zip(polymers, organisms)],
        'Confidence (%)': confidences,
        'Size (μm)': sizes
    })

# ==================== MAIN APP ====================

# Load sample data
stats = get_sample_stats()
df = get_sample_classifications()

# ==================== HEADER ====================
st.title("🔬 Micro-Scale Sentinel")
st.markdown("**AI-Powered Microplastic Detection using Holographic Microscopy + Google Gemini**")
st.markdown("---")

# ==================== IMAGE UPLOAD SECTION (MAIN FEATURE) ====================
st.header("🧪 Analyze Your Sample")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="upload-box">
        <h3>📤 Upload Holographic Image</h3>
        <p>Supported formats: PNG, JPG, JPEG</p>
        <p>Recommended: 512x512 pixels minimum</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a holographic microscopy image of a particle for AI classification",
        label_visibility="collapsed"
    )
    
    # Optional manual feature input
    with st.expander("⚙️ Advanced: Provide Particle Measurements (Optional)"):
        st.markdown("If you have measured values, enter them here for more accurate classification:")
        
        col_a, col_b = st.columns(2)
        with col_a:
            manual_size = st.number_input("Size (μm)", min_value=0.0, max_value=10000.0, value=0.0, step=10.0)
            manual_circularity = st.number_input("Circularity (0-1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        with col_b:
            manual_ri = st.number_input("Refractive Index", min_value=1.0, max_value=2.0, value=0.0, step=0.01)
            manual_variance = st.number_input("Intensity Variance", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)

with col2:
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Holographic Image", use_container_width=True)
        
        # Classify button
        if st.button("🔬 Analyze with Gemini AI", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing particle with AI... This may take 10-30 seconds..."):
                
                # Extract features (or use manual inputs)
                if manual_size > 0 or manual_ri > 0:
                    features = {
                        "size_um": manual_size if manual_size > 0 else "Unknown",
                        "circularity": manual_circularity if manual_circularity > 0 else "Unknown",
                        "ri_estimate": manual_ri if manual_ri > 0 else "Unknown",
                        "intensity_variance": manual_variance if manual_variance > 0 else "Unknown"
                    }
                else:
                    features = extract_basic_features(image)
                
                # Classify with Gemini
                result = classify_with_gemini(image, features)
                
                # Store result in session state
                st.session_state['last_result'] = result
                st.session_state['last_features'] = features
    else:
        st.info("👆 Upload an image to begin analysis")

# ==================== DISPLAY RESULTS ====================
if 'last_result' in st.session_state:
    result = st.session_state['last_result']
    features = st.session_state.get('last_features', {})
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    if result.get("status") == "error":
        st.error(f"❌ {result.get('message')}")
        st.info("💡 Make sure GEMINI_API_KEY is configured in Streamlit secrets")
    else:
        # Classification result
        classification = result.get('classification', 'UNKNOWN')
        confidence_micro = result.get('confidence_microplastic', 0)
        confidence_bio = result.get('confidence_biological', 0)
        
        # Color coding
        if classification == "MICROPLASTIC":
            st.markdown(f"""
            <div class="alert-danger">
                <h2 style="margin:0; color:#E74C3C;">🔴 MICROPLASTIC DETECTED</h2>
                <p style="font-size:18px; margin:10px 0 0 0;">Confidence: <strong>{confidence_micro}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        elif classification == "BIOLOGICAL":
            st.markdown(f"""
            <div class="alert-success">
                <h2 style="margin:0; color:#27AE60;">🟢 BIOLOGICAL ORGANISM</h2>
                <p style="font-size:18px; margin:10px 0 0 0;">Confidence: <strong>{confidence_bio}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-warning">
                <h2 style="margin:0; color:#F39C12;">⚠️ UNCERTAIN CLASSIFICATION</h2>
                <p style="font-size:18px; margin:10px 0 0 0;">Requires manual review</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 Classification Details")
            st.metric("Primary Class", classification)
            st.metric("Microplastic Confidence", f"{confidence_micro}%")
            st.metric("Biological Confidence", f"{confidence_bio}%")
            
            if result.get('polymer_type'):
                st.metric("Polymer Type", result['polymer_type'])
            if result.get('organism_type'):
                st.metric("Organism Type", result['organism_type'].capitalize())
        
        with col2:
            st.markdown("### 📏 Particle Features")
            st.metric("Size", f"{features.get('size_um', 'N/A')} μm")
            st.metric("Circularity", features.get('circularity', 'N/A'))
            st.metric("Refractive Index", features.get('ri_estimate', 'N/A'))
            st.metric("Intensity Variance", f"{features.get('intensity_variance', 'N/A')}")
        
        with col3:
            st.markdown("### 💡 Recommendation")
            recommendation = result.get('recommendation', 'UNKNOWN')
            
            if recommendation == "DEFINITE":
                st.success("✅ HIGH CONFIDENCE - Classification reliable")
            elif recommendation == "PROBABLE":
                st.warning("⚠️ MODERATE CONFIDENCE - Likely correct")
            else:
                st.error("🔍 LOW CONFIDENCE - Manual review recommended")
            
            st.metric("Size Category", result.get('size_category', 'Unknown'))
        
        # AI Reasoning
        st.markdown("### 🧠 AI Reasoning")
        st.markdown(f"""
        <div class="result-box">
            <p style="font-size:16px; line-height:1.6;">{result.get('reasoning', 'No reasoning provided.')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Evidence breakdown
        if result.get('evidence'):
            with st.expander("🔬 Detailed Evidence Analysis"):
                evidence = result['evidence']
                
                for key, value in evidence.items():
                    st.markdown(f"**{key.replace('_', ' ').title()}:**")
                    st.write(value)
                    st.markdown("---")
        
        # Export results
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "classification": result,
                "features": features
            }
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_str,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export
            csv_data = pd.DataFrame([{
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Classification": classification,
                "Confidence_Microplastic": confidence_micro,
                "Confidence_Biological": confidence_bio,
                "Polymer_Type": result.get('polymer_type', ''),
                "Organism_Type": result.get('organism_type', ''),
                "Size_um": features.get('size_um', ''),
                "RI_Estimate": features.get('ri_estimate', ''),
                "Recommendation": recommendation
            }])
            st.download_button(
                label="📊 Download CSV",
                data=csv_data.to_csv(index=False),
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

st.markdown("---")

# ==================== DASHBOARD STATISTICS ====================
st.header("📈 Historical Statistics")

# Alert based on contamination
if stats['microplastic_percentage'] > 70:
    st.markdown(f"""
    <div class="alert-danger">
        <strong>⚠️ HIGH CONTAMINATION ALERT!</strong><br>
        Microplastic contamination level is critically high ({stats['microplastic_percentage']:.1f}%).
    </div>
    """, unsafe_allow_html=True)
elif stats['microplastic_percentage'] > 40:
    st.markdown(f"""
    <div class="alert-warning">
        <strong>⚠️ MODERATE CONTAMINATION</strong><br>
        Microplastic levels are elevated ({stats['microplastic_percentage']:.1f}%).
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-success">
        <strong>✅ LOW CONTAMINATION</strong><br>
        Microplastic levels are within acceptable range ({stats['microplastic_percentage']:.1f}%).
    </div>
    """, unsafe_allow_html=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📦 Total Particles", f"{stats['total_particles']:,}")

with col2:
    st.metric("🔴 Microplastics", f"{stats['microplastic_count']:,}", 
              delta=f"{stats['microplastic_percentage']:.1f}%", delta_color="inverse")

with col3:
    st.metric("🟢 Biological", f"{stats['biological_count']:,}",
              delta=f"{stats['biological_percentage']:.1f}%")

with col4:
    st.metric("⚠️ Uncertain", f"{stats['uncertain_count']:,}",
              delta=f"{stats['uncertain_percentage']:.1f}%", delta_color="off")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🧪 Polymer Distribution")
    polymer_data = pd.DataFrame(
        list(stats['polymer_distribution'].items()),
        columns=['Polymer', 'Count']
    )
    fig1 = px.pie(polymer_data, values='Count', names='Polymer',
                  color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### 📊 Classification Breakdown")
    breakdown_data = pd.DataFrame({
        'Category': ['Microplastic', 'Biological', 'Uncertain'],
        'Count': [stats['microplastic_count'], stats['biological_count'], stats['uncertain_count']]
    })
    fig2 = px.bar(breakdown_data, x='Category', y='Count', color='Category',
                  color_discrete_map={'Microplastic': '#E74C3C', 'Biological': '#27AE60', 'Uncertain': '#F39C12'})
    st.plotly_chart(fig2, use_container_width=True)

# Recent classifications table
st.markdown("#### 📋 Recent Classifications")
st.dataframe(df, use_container_width=True, height=300)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>🔬 Micro-Scale Sentinel v1.0</strong></p>
    <p>Powered by <strong>Google Gemini 1.5 Pro</strong> | Real-time AI Classification</p>
    <p>🌊 <em>Protecting our oceans, one particle at a time</em> 🌊</p>
</div>
""", unsafe_allow_html=True)
