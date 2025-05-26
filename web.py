import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import lief
import traceback
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
# Import the feature extraction function from app.py
from app import extract_pe_features

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load and analyze model results
@st.cache_data
def load_model_analysis():
    # Load model and metadata
    logger.debug("Loading model and metadata...")
    model_info = joblib.load('random_forest_model.joblib')
    model = model_info['model']
    optimal_threshold = model_info['optimal_threshold']
    feature_names = model_info['feature_names']
    scaler = model_info['scaler']
    logger.debug(f"Model loaded. Threshold: {optimal_threshold}")
    
    # Load test data for visualization
    df = pd.read_csv('malware_detection_features.csv')
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Scale features
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    return {
        'model': model,
        'optimal_threshold': optimal_threshold,
        'feature_names': feature_names,
        'scaler': scaler,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'importances': importances,
        'indices': indices,
        'confusion_matrix': confusion_matrix(y, y_pred)
    }

# Set page configuration
st.set_page_config(
    page_title="Malware Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model analysis
analysis_results = load_model_analysis()
model = analysis_results['model']
optimal_threshold = analysis_results['optimal_threshold']
feature_names = analysis_results['feature_names']
scaler = analysis_results['scaler']

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stAlert { padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .result-card { 
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .malware { background-color: #dc3545; color: white; }
    .benign { background-color: #198754; color: white; }    .metric-card {
        background-color: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🛡️ Malware Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #7f8c8d;'>Random Forest Model Analysis & File Scanner</p>", unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2 = st.tabs(["📊 Model Analysis", "🔍 File Scanner"])

with tab1:
    st.markdown("## Model Analysis Results")
    
    # Model metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Trees", model.n_estimators)
    with col2:
        st.metric("Max Depth", model.max_depth)
    with col3:
        st.metric("Default Threshold", f"{optimal_threshold:.3f}")
    
    # Feature importance plot
    st.markdown("### Feature Importance")
    fig, ax = plt.subplots(figsize=(12, 6))
    n_features = min(10, len(analysis_results['importances']))
    plt.bar(range(n_features), 
            analysis_results['importances'][analysis_results['indices'][:n_features]])
    plt.xticks(range(n_features), 
               [feature_names[i] for i in analysis_results['indices'][:n_features]], 
               rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Top 10 Important Features')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Confusion matrix
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(analysis_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)

with tab2:
    st.markdown("## File Scanner")
    
    # Add threshold slider
    custom_threshold = st.slider(
        "Detection Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=optimal_threshold,
        help="Adjust this value to control the sensitivity of malware detection. Higher values mean stricter detection."
    )
    
    uploaded_file = st.file_uploader("Choose a PE file for analysis", type=["exe", "dll"])
    
    if uploaded_file:
        with st.spinner("Analyzing file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            try:
                # Extract features using our custom function
                logger.debug(f"Extracting features from file: {file_path}")
                features_df = extract_pe_features(file_path)
                logger.debug(f"Features extracted: {features_df.shape[0]} samples")
                logger.debug(f"Feature names: {list(features_df.columns)}")
                
                # Scale features
                features_scaled = scaler.transform(features_df)
                logger.debug(f"Features scaled. Shape: {features_scaled.shape}")
                
                # Get prediction and probability
                probability = model.predict_proba(features_scaled)[0][1]
                prediction = int(probability >= custom_threshold)
                logger.debug(f"Raw probability: {probability:.4f}, Threshold: {custom_threshold}")
                logger.debug(f"Final prediction: {prediction}")
                
                # Display result with more detail
                result_color = "#dc3545" if prediction == 1 else "#198754"
                result_text = "⚠️ Potential Malware" if prediction == 1 else "✅ Likely Benign"
                
                # Show detailed probability score
                st.markdown(f"""
                    <div class='result-card' style='background-color: {result_color};'>
                        <h2>{result_text}</h2>
                        <p style='font-size: 1.2em;'>Malware Probability: {probability:.2%}</p>
                        <p>Using threshold: {custom_threshold:.3f}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add probability gauge
                st.markdown("### Confidence Meter")
                st.progress(probability)
                
                # Feature contribution
                st.markdown("### Most Influential Features")
                feature_contribs = features_scaled[0] * analysis_results['importances']
                top_contribs = sorted(zip(feature_names, feature_contribs), 
                                    key=lambda x: abs(x[1]), reverse=True)[:5]
                
                # Show raw feature values
                st.markdown("### Raw Feature Values")
                for feat, value in features_df.iloc[0].items():
                    st.markdown(f"""
                    <div class='metric-card'>
                        <strong>{feat}</strong>: {value:.3f}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show feature contributions
                st.markdown("### Feature Contributions")
                for feat, contrib in top_contribs:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <strong>{feat}</strong>: {contrib:.3f}
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing file: {str(e)}")
                logger.error(f"Error analyzing file {file_path}: {str(e)}", exc_info=True)
            finally:
                os.unlink(file_path)