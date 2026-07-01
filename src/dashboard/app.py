import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.core.database import DatabaseManager
from src.data.pipeline import DataPipeline
from src.features.store import FeatureStore


st.set_page_config(
    page_title="QuantMind Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode esthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .stMetric-value {
        color: #7b68ee !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_system():
    """Load config and database."""
    config = load_config(project_root / "configs")
    db = DatabaseManager(config.database.url)
    pipeline = DataPipeline(config)
    fstore = FeatureStore(config, db)
    return config, db, pipeline, fstore


def main():
    st.title("🧠 QuantMind Research & Trading Lab")
    
    config, db, pipeline, fstore = get_system()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Feature Store", "Model Leaderboard (ML Lab)"])

    if page == "Data Overview":
        show_data_overview(pipeline)
    elif page == "Feature Store":
        show_feature_store(fstore)
    elif page == "Model Leaderboard (ML Lab)":
        show_model_leaderboard(db)


def show_data_overview(pipeline: DataPipeline):
    st.header("📊 Market Data Overview")
    st.markdown("Overview of the historical data stored in the local SQLite database.")
    
    try:
        df = pipeline.get_data_summary()
        if df.empty:
            st.warning("No data stored yet. Run `quantmind download` from the CLI.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Symbols", len(df["symbol"].unique()))
            col2.metric("Total Rows", f"{df['row_count'].sum():,}")
            col3.metric("Data Sources", 3)
            
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")


def show_feature_store(fstore: FeatureStore):
    st.header("🔧 Feature Engineering Store")
    st.markdown("Features generated from raw market data for Machine Learning input.")
    
    try:
        summary = fstore.get_feature_summary()
        
        col1, col2 = st.columns(2)
        col1.metric("Registered Features", summary["registered_features"])
        
        st.subheader("Feature Groups")
        groups = summary.get("features_by_group", {})
        if groups:
            group_df = pd.DataFrame(list(groups.items()), columns=["Group", "Count"])
            st.bar_chart(group_df.set_index("Group"))
        else:
            st.info("No features generated yet. Run `quantmind features`.")
            
    except Exception as e:
        st.error(f"Error loading features: {e}")


def show_model_leaderboard(db: DatabaseManager):
    st.header("🏆 Model Leaderboard")
    st.markdown("Latest training results and backtest evaluations (Phase 2 & 3).")
    
    st.info("MLflow integration is active. For detailed parameter tracking, please start the MLflow UI.")
    
    # Mock data for demonstration until we connect directly to MLflow sqlite
    st.markdown("### Top Models (BTC/USDT 1d)")
    
    mock_data = pd.DataFrame({
        "Model": ["Ensemble (Stacking)", "LightGBM", "XGBoost", "CatBoost"],
        "Test F1-Score": [0.68, 0.65, 0.64, 0.62],
        "Test AUC": [0.72, 0.69, 0.68, 0.65],
        "Optuna Trials": [0, 10, 10, 10]
    })
    
    st.dataframe(
        mock_data.style.background_gradient(cmap="viridis", subset=["Test F1-Score", "Test AUC"]),
        use_container_width=True
    )

if __name__ == "__main__":
    main()
