# filepath: c:\Users\thinkbook\Desktop\ember_dataset_2018_2\app.py
from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from joblib import load
import lief
import json
import os
import tempfile
import re
import math

app = Flask(__name__)

# Load the trained model and metadata
print("Loading model...")
model_info = load('random_forest_model.joblib')
model = model_info['model']
optimal_threshold = model_info['optimal_threshold']
feature_names = model_info['feature_names']
scaler = model_info['scaler']

def extract_pe_features(file_path):
    """Extract features from PE file using LIEF"""
    try:
        # Parse PE file
        binary = lief.parse(file_path)
        if binary is None:
            raise ValueError("Failed to parse PE file")
        
        # Initialize feature dictionary
        features = {}
        
        # Basic file features
        features['file_size'] = os.path.getsize(file_path)
        features['virtual_size'] = sum(section.virtual_size for section in binary.sections)
        features['size_ratio'] = features['virtual_size'] / features['file_size'] if features['file_size'] > 0 else 0
        
        # Section analysis
        suspicious_section_names = {'.text', '.data', '.rdata', '.idata', '.edata', '.pdata', '.rsrc'}
        features['suspicious_sections'] = sum(1 for s in binary.sections if s.name not in suspicious_section_names)
        
        # Import analysis
        suspicious_imports = {'kernel32.dll', 'user32.dll', 'advapi32.dll', 'ws2_32.dll', 'wininet.dll', 'shell32.dll'}
        if binary.has_imports:
            features['suspicious_imports'] = sum(1 for imp in binary.imports if imp.name.lower() in suspicious_imports)
        else:
            features['suspicious_imports'] = 0
        
        # Entropy features
        features['max_section_entropy'] = max(s.entropy for s in binary.sections)
        
        # String analysis
        all_strings = []
        for section in binary.sections:
            try:
                content = section.content
                strings = [s for s in content if len(s) >= 4 and s.isprintable()]
                all_strings.extend(strings)
            except:
                continue
        features['string_entropy'] = calculate_entropy(''.join(str(s) for s in all_strings)) if all_strings else 0
        
        # URL and special patterns
        strings_str = ''.join(str(s) for s in all_strings)
        features['num_urls'] = len(re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', strings_str))
        features['num_registry'] = len(re.findall(r'HKEY_|HKLM|HKCU|HKCR|HKU|HKCC', strings_str))
        features['num_MZ'] = len(re.findall(b'MZ', binary.content)) if hasattr(binary, 'content') else 0
        
        # Create DataFrame with proper feature names
        df = pd.DataFrame([features])
        
        # Ensure all features from training are present
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0
                
        # Select only the features used in training
        df = df[feature_names]
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error extracting features: {str(e)}")

def calculate_entropy(data):
    """Calculate Shannon entropy of a string"""
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(chr(x)))/len(data)
        if p_x > 0:
            entropy += - p_x * math.log2(p_x)
    return entropy

def predict_malware(X_new, threshold=None):
    """Make predictions using the Random Forest model"""
    if threshold is None:
        threshold = optimal_threshold
    
    # Scale features
    X_scaled = scaler.transform(X_new)
    
    # Get probabilities
    proba = model.predict_proba(X_scaled)[:, 1]
    
    # Make predictions using threshold
    predictions = (proba >= threshold).astype(int)
    
    return predictions, proba

@app.route('/')
def home():
    # Get feature importance data
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = []
    n_top = min(10, len(importances))
    for i in range(n_top):
        top_features.append({
            'name': feature_names[indices[i]],
            'importance': float(importances[indices[i]])
        })
    
    return render_template('index.html', 
                         feature_names=feature_names,
                         top_features=top_features,
                         threshold=float(optimal_threshold))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as tmp_file:
            file.save(tmp_file.name)
            file_path = tmp_file.name
            
        try:
            # Extract features
            features_df = extract_pe_features(file_path)
            
            # Make prediction
            predictions, probabilities = predict_malware(features_df)
            
            # Get feature contributions
            features_scaled = scaler.transform(features_df)
            feature_importance = model.feature_importances_
            feature_contributions = features_scaled[0] * feature_importance
            
            # Get top contributing features
            top_contributors = sorted(
                zip(feature_names, feature_contributions),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            return jsonify({
                'status': 'success',
                'prediction': int(predictions[0]),
                'probability': float(probabilities[0]),
                'threshold': float(optimal_threshold),
                'top_contributors': [
                    {'feature': feat, 'contribution': float(cont)}
                    for feat, cont in top_contributors
                ]
            })
            
        finally:
            # Clean up temporary file
            os.unlink(file_path)
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)