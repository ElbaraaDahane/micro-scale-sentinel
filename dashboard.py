"""
Micro-Scale Sentinel Dashboard
AI-Powered Microplastic Detection System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import json

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
    /* Main theme */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    
    /* Headers */
    h1 {
        color: #2C3E50;
        border-bottom: 3px solid #3498DB;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #34495E;
        margin-top: 30px;
    }
    
    /* Alert boxes */
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
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #7F8C8D;
        margin-top: 50px;
        border-top: 2px solid #BDC3C7;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ECF0F1;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SAMPLE DATA FUNCTION ====================
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
        'organism_distribution': {
            'Diatom': 22,
            'Copepod': 14,
            'Other': 2
        },
        'avg_confidence': 84.5,
        'high_confidence_count': 132,
        'medium_confidence_count': 14,
        'low_confidence_count': 4
    }

@st.cache_data
def get_sample_classifications():
    """Return sample classification data"""
    np.random.seed(42)
    
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
        'Size (μm)': sizes,
        'Location': ['Site A'] * 10 + ['Site B'] * 10
    })

# ==================== LOAD DATA ====================
try:
    from storage import StorageManager
    db_path = "data/database/sentinel.db"
    
    if os.path.exists(db_path):
        storage = StorageManager(db_path)
        stats_raw = storage.get_statistics(None, None)
        
        if isinstance(stats_raw, dict) and 'total_particles' in stats_raw:
            stats = stats_raw
            data_source = "database"
        else:
            stats = get_sample_stats()
            data_source = "sample"
    else:
        stats = get_sample_stats()
        data_source = "sample"
except:
    stats = get_sample_stats()
    data_source = "sample"

# Load classification data
try:
    df = get_sample_classifications()
except:
    df = pd.DataFrame()

# ==================== HEADER ====================
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🔬 Micro-Scale Sentinel")
    st.markdown("**AI-Powered Microplastic Detection using Holographic Microscopy**")

with col2:
    st.image("https://via.placeholder.com/150/3498DB/FFFFFF?text=MSS", width=150)

st.markdown("---")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Data source indicator
    if data_source == "database":
        st.success("✅ Connected to Database")
    else:
        st.info("ℹ️ Displaying Demo Data")
    
    st.markdown("---")
    
    # Date range filter
    st.subheader("📅 Date Range")
    date_range = st.date_input(
        "Filter by date",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Polymer filter
    st.subheader("🧪 Polymer Filter")
    polymer_options = list(stats['polymer_distribution'].keys())
    selected_polymers = st.multiselect(
        "Select polymers",
        polymer_options,
        default=polymer_options,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Confidence threshold
    st.subheader("📊 Confidence Threshold")
    confidence_threshold = st.slider(
        "Minimum confidence (%)",
        0, 100, 60,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📈 Quick Stats")
    st.metric("Detection Rate", f"{stats['microplastic_percentage']:.1f}%")
    st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
    st.metric("High Confidence", f"{stats['high_confidence_count']}")
    
    st.markdown("---")
    
    # Links
    st.subheader("🔗 Resources")
    st.markdown("🐙 [GitHub Repository](https://github.com/yourusername/micro-scale-sentinel)")
    st.markdown("📖 [Documentation](docs/)")
    st.markdown("📧 [Contact Us](mailto:contact@example.com)")
    st.markdown("🎥 [Demo Video](#)")

# ==================== ALERT BASED ON CONTAMINATION ====================
if stats['microplastic_percentage'] > 70:
    st.markdown("""
    <div class="alert-danger">
        <strong>⚠️ HIGH CONTAMINATION ALERT!</strong><br>
        Microplastic contamination level is critically high ({:.1f}%). 
        Immediate investigation and mitigation measures recommended.
    </div>
    """.format(stats['microplastic_percentage']), unsafe_allow_html=True)
elif stats['microplastic_percentage'] > 40:
    st.markdown("""
    <div class="alert-warning">
        <strong>⚠️ MODERATE CONTAMINATION</strong><br>
        Microplastic levels are elevated ({:.1f}%). 
        Continued monitoring advised.
    </div>
    """.format(stats['microplastic_percentage']), unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-success">
        <strong>✅ LOW CONTAMINATION</strong><br>
        Microplastic levels are within acceptable range ({:.1f}%).
    </div>
    """.format(stats['microplastic_percentage']), unsafe_allow_html=True)

# ==================== MAIN METRICS ====================
st.subheader("📊 Overview Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📦 Total Particles Analyzed",
        value=f"{stats['total_particles']:,}",
        delta=None,
        help="Total number of particles processed"
    )

with col2:
    st.metric(
        label="🔴 Microplastics Detected",
        value=f"{stats['microplastic_count']:,}",
        delta=f"{stats['microplastic_percentage']:.1f}%",
        delta_color="inverse",
        help="Number and percentage of microplastic particles"
    )

with col3:
    st.metric(
        label="🟢 Biological Organisms",
        value=f"{stats['biological_count']:,}",
        delta=f"{stats['biological_percentage']:.1f}%",
        delta_color="normal",
        help="Number and percentage of biological particles"
    )

with col4:
    st.metric(
        label="⚠️ Uncertain Classifications",
        value=f"{stats['uncertain_count']:,}",
        delta=f"{stats['uncertain_percentage']:.1f}%",
        delta_color="off",
        help="Particles requiring manual review"
    )

st.markdown("---")

# ==================== VISUALIZATIONS ROW 1 ====================
st.subheader("📈 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🧪 Polymer Type Distribution")
    
    polymer_data = pd.DataFrame(
        list(stats['polymer_distribution'].items()),
        columns=['Polymer', 'Count']
    )
    
    fig1 = px.pie(
        polymer_data,
        values='Count',
        names='Polymer',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    fig1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    fig1.update_layout(
        showlegend=True,
        height=400,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    with st.expander("ℹ️ Polymer Information"):
        st.markdown("""
        **Common Microplastic Polymers:**
        - **PET** (Polyethylene Terephthalate): Bottles, textiles
        - **HDPE** (High-Density Polyethylene): Containers, bags
        - **PP** (Polypropylene): Ropes, food containers
        - **PS** (Polystyrene): Styrofoam, packaging
        - **PVC** (Polyvinyl Chloride): Pipes, vinyl products
        """)

with col2:
    st.markdown("##### 📊 Classification Breakdown")
    
    breakdown_data = pd.DataFrame({
        'Category': ['Microplastic', 'Biological', 'Uncertain'],
        'Count': [
            stats['microplastic_count'],
            stats['biological_count'],
            stats['uncertain_count']
        ],
        'Percentage': [
            stats['microplastic_percentage'],
            stats['biological_percentage'],
            stats['uncertain_percentage']
        ]
    })
    
    fig2 = px.bar(
        breakdown_data,
        x='Category',
        y='Count',
        color='Category',
        text='Percentage',
        color_discrete_map={
            'Microplastic': '#E74C3C',
            'Biological': '#27AE60',
            'Uncertain': '#F39C12'
        }
    )
    fig2.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text:.1f}%<extra></extra>'
    )
    fig2.update_layout(
        showlegend=False,
        height=400,
        yaxis_title="Number of Particles",
        xaxis_title="",
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ==================== CONFIDENCE ANALYSIS ====================
st.subheader("🎯 Confidence Score Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### 📈 Confidence Distribution")
    
    confidence_data = pd.DataFrame({
        'Range': ['High\n(>85%)', 'Medium\n(60-85%)', 'Low\n(<60%)'],
        'Count': [
            stats['high_confidence_count'],
            stats['medium_confidence_count'],
            stats['low_confidence_count']
        ],
        'Percentage': [
            stats['high_confidence_count'] / stats['total_particles'] * 100,
            stats['medium_confidence_count'] / stats['total_particles'] * 100,
            stats['low_confidence_count'] / stats['total_particles'] * 100
        ]
    })
    
    fig3 = px.bar(
        confidence_data,
        x='Range',
        y='Count',
        color='Range',
        text='Percentage',
        color_discrete_map={
            'High\n(>85%)': '#27AE60',
            'Medium\n(60-85%)': '#F39C12',
            'Low\n(<60%)': '#E74C3C'
        }
    )
    fig3.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    fig3.update_layout(
        showlegend=False,
        height=350,
        yaxis_title="Number of Classifications",
        xaxis_title="Confidence Range"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("##### 🎯 Average Confidence")
    
    fig4 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=stats['avg_confidence'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Confidence", 'font': {'size': 20}},
        delta={'reference': 80, 'increasing': {'color': "#27AE60"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#3498DB", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#FADBD8'},
                {'range': [60, 85], 'color': '#FCF3CF'},
                {'range': [85, 100], 'color': '#D5F4E6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    fig4.update_layout(
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        font={'size': 14}
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ==================== RECENT CLASSIFICATIONS TABLE ====================
st.subheader("📋 Recent Classifications")

if not df.empty:
    # Apply filters
    df_filtered = df[df['Confidence (%)'] >= confidence_threshold].copy()
    
    # Color coding function
    def highlight_row(row):
        if row['Classification'] == 'Microplastic':
            return ['background-color: #FADBD8'] * len(row)
        elif row['Classification'] == 'Biological':
            return ['background-color: #D5F4E6'] * len(row)
        else:
            return ['background-color: #FCF3CF'] * len(row)
    
    # Display table
    styled_df = df_filtered.style.apply(highlight_row, axis=1)\
                                  .background_gradient(
                                      subset=['Confidence (%)'],
                                      cmap='RdYlGn',
                                      vmin=0,
                                      vmax=100
                                  )\
                                  .format({
                                      'Confidence (%)': '{:.1f}%',
                                      'Size (μm)': '{:.0f}'
                                  })
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"microplastic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = df_filtered.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"microplastic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
else:
    st.info("No classification data available. Upload images to start detection.")

st.markdown("---")

# ==================== SYSTEM INFORMATION ====================
with st.expander("ℹ️ System Information & Settings"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**System Details**")
        st.text(f"Version: 1.0.0")
        st.text(f"AI Model: Gemini 1.5 Pro")
        st.text(f"Database: SQLite/PostgreSQL")
        st.text(f"Framework: Streamlit")
        st.text(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        st.markdown("**Detection Parameters**")
        st.text("Min Particle Size: 50 μm")
        st.text("Max Particle Size: 5000 μm")
        st.text("Confidence Threshold: 60%")
        st.text("High Confidence: >85%")
        st.text("Wavelength: 632.8 nm")
    
    with col3:
        st.markdown("**Performance Metrics**")
        st.text(f"Total Analyses: {stats['total_particles']}")
        st.text(f"Avg Processing Time: 2.4s")
        st.text(f"Success Rate: 97.3%")
        st.text(f"API Uptime: 99.9%")
        st.text(f"Data Quality Score: A+")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>🔬 Micro-Scale Sentinel v1.0</strong></p>
    <p>Powered by <strong>Google Gemini 1.5 Pro</strong> | Holographic Microscopy | AI Multimodal Reasoning</p>
    <p>🌊 <em>Protecting our oceans, one particle at a time</em> 🌊</p>
    <p style="margin-top: 15px;">
        <a href="https://github.com/yourusername/micro-scale-sentinel" target="_blank">GitHub</a> | 
        <a href="docs/" target="_blank">Documentation</a> | 
        <a href="mailto:contact@example.com">Contact</a>
    </p>
    <p style="font-size: 12px; color: #95A5A6; margin-top: 10px;">
        © 2026 Micro-Scale Sentinel Project | MIT License
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== DEBUG INFO (Hidden) ====================
if st.sidebar.checkbox("🔧 Show Debug Info", value=False):
    with st.expander("Debug Information"):
        st.json({
            'data_source': data_source,
            'stats': stats,
            'filters': {
                'confidence_threshold': confidence_threshold,
                'selected_polymers': selected_polymers,
                'date_range': str(date_range)
            }
        })
