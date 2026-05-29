import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# Set page configuration
st.set_page_config(page_title="🌳 Forest Carbon MRV", layout="wide")

st.title("🌳 Forest Carbon MRV")
st.markdown("Monitor Forest carbon stocks using satellite imagery and AI.")

# Sidebar for users' input
with st.sidebar:
    st.header("Select Region")
    region = st.selectbox("Region", ['Amazon (Brazil)', "Southeast Asia", "Central Africa"])
    st.header("Time Period")
    year = st.selectbox("Year", [2023, 2024, 2025], index=1)
    st.header("Show Explainability (Grad-CAM)")
    show_gradcam = st.checkbox("Show model attention map.")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Forest Cover Map")
    # Query Athena for the latest map, now we'll create a simple map
    m = folium.Map(location=[-3.465, -62.215], zoom_start=6) # Amz coordinates
    # Add a marker or simple rectangle for demo
    folium.Rectangle(bounds=[(-4.0, -63.0), (-2.5, -61.0)], color="green", 
                     fill=True, popup="Forest Area").add_to(m)
    folium_static(m)

with col2:
    st.subheader("Carbon Stock Estimates")
    # Sample data - replace with real data from Athena later
    data = {
        "Region": ['Amazon', 'Southeast Asia', 'Central Africa'],
        "Forest Area (ha)": [350000, 200000, 180000],
        "Carbon Stock (tCO2e)": [5250000, 3000000, 2700000]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

    st.subheader("Change over time")
    # Sample time series
    years = [2023, 2024, 2025]
    carbon = [5.2e6, 5.1e6, 4.9e6] # in tCO2e
    fig, ax = plt.subplots()
    ax.plot(years, carbon, marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Carbon Stock (tCO2e)")
    ax.set_title("Carbon Trends")
    st.pyplot(fig)

if show_gradcam:
    st.subheader("Model Attention (Grad-CAM)")
    st.image("https://via.placeholder.com/800x400?text=Grad-CAM+Heatmap", caption="Example attention map")
    st.markdown("This shows where the model focused most to clarify forest pixels.")