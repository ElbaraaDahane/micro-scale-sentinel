
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from storage import StorageManager

# Page config
st.set_page_config(
    page_title="Micro-Scale Sentinel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-metric { font-size: 24px; font-weight: bold; }
.stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔬 Micro-Scale Sentinel")
st.markdown("**AI-Powered Microplastic Detection System**")
st.markdown("---")

# Initialize storage (use sample database or create empty)
try:
    storage = StorageManager("data/database/sentinel.db")
    stats = storage.get_statistics(None, None)
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.info("Creating sample database...")
    # Create sample data for demo
    stats = {
        'total_particles': 150,
        'microplastic_count': 108,
        'biological_count': 38,
        'uncertain_count': 4,
        'microplastic_percentage': 72.0,
        'polymer_distribution': {
            'PET': 45,
            'HDPE': 28,
            'PP': 20,
            'PS': 10,
            'PVC': 5
        },
        'avg_confidence': 84.5
    }

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now())
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Detection Rate", f"{stats['microplastic_percentage']:.1f}%")
    st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
    
    st.markdown("---")
    st.markdown("### 📚 Documentation")
    st.markdown("[Technical Architecture](docs/technical-architecture.md)")
    st.markdown("[Prompt Engineering](docs/prompt-engineering-guide.md)")
    st.markdown("[Deployment Guide](docs/deployment-instructions.md)")

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📦 Total Particles",
        value=f"{stats['total_particles']:,}",
        delta=None
    )

with col2:
    st.metric(
        label="🔴 Microplastics",
        value=f"{stats['microplastic_count']:,}",
        delta=f"{stats['microplastic_percentage']:.1f}%"
    )

with col3:
    st.metric(
        label="🟢 Biological",
        value=f"{stats['biological_count']:,}",
        delta=None
    )

with col4:
    st.metric(
        label="⚠️ Uncertain",
        value=f"{stats['uncertain_count']:,}",
        delta=None
    )

st.markdown("---")

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Polymer Type Distribution")
    polymer_data = pd.DataFrame(
        list(stats['polymer_distribution'].items()),
        columns=['Polymer', 'Count']
    )
    fig1 = px.pie(
        polymer_data,
        values='Count',
        names='Polymer',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📊 Classification Breakdown")
    breakdown_data = pd.DataFrame({
        'Category': ['Microplastic', 'Biological', 'Uncertain'],
        'Count': [
            stats['microplastic_count'],
            stats['biological_count'],
            stats['uncertain_count']
        ]
    })
    fig2 = px.bar(
        breakdown_data,
        x='Category',
        y='Count',
        color='Category',
        color_discrete_map={
            'Microplastic': '#FF6B6B',
            'Biological': '#4ECDC4',
            'Uncertain': '#FFE66D'
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Recent classifications table
st.subheader("📋 Recent Classifications")

# Sample data (replace with actual database query in production)
sample_data = pd.DataFrame({
    'Timestamp': pd.date_range(end=datetime.now(), periods=10, freq='H')[::-1],
    'Particle ID': [f'P{i:03d}' for i in range(1, 11)],
    'Classification': ['Microplastic'] * 7 + ['Biological'] * 2 + ['Uncertain'],
    'Polymer/Organism': ['PET', 'HDPE', 'PP', 'PET', 'PS', 'HDPE', 'PVC', 'Diatom', 'Copepod', '-'],
    'Confidence (%)': [87, 92, 78, 85, 91, 76, 88, 89, 94, 45],
    'Size (μm)': [245, 312, 189, 267, 198, 334, 156, 128, 445, 289]
})

st.dataframe(
    sample_data.style.background_gradient(subset=['Confidence (%)'], cmap='RdYlGn'),
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Micro-Scale Sentinel v1.0</strong></p>
    <p>Powered by Google Gemini 1.5 Pro | Holographic Microscopy | AI Classification</p>
    <p>📧 Contact | 📖 <a href="https://github.com/yourusername/micro-scale-sentinel">GitHub</a> | 📄 Documentation</p>
</div>
""", unsafe_allow_html=True)
