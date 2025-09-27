"""
Data Preprocessing Module for MatrixLab AI Studio
Handles data cleaning, encoding, and transformation
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder, Normalizer
)
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles comprehensive data preprocessing operations"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
    
    def handle_missing_values(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Handle missing values using specified method"""
        data_copy = data.copy()
        
        if method == "Drop rows with missing values":
            original_shape = data_copy.shape
            data_copy = data_copy.dropna()
            st.info(f"Dropped {original_shape[0] - data_copy.shape[0]} rows with missing values")
        
        elif method == "Fill with mean (numeric)":
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data_copy[col].isnull().any():
                    mean_val = data_copy[col].mean()
                    data_copy[col].fillna(mean_val, inplace=True)
                    st.info(f"Filled {col} missing values with mean: {mean_val:.2f}")
        
        elif method == "Fill with median (numeric)":
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data_copy[col].isnull().any():
                    median_val = data_copy[col].median()
                    data_copy[col].fillna(median_val, inplace=True)
                    st.info(f"Filled {col} missing values with median: {median_val:.2f}")
        
        elif method == "Fill with mode":
            for col in data_copy.columns:
                if data_copy[col].isnull().any():
                    mode_value = data_copy[col].mode()
                    if len(mode_value) > 0:
                        data_copy[col].fillna(mode_value[0], inplace=True)
                        st.info(f"Filled {col} missing values with mode: {mode_value[0]}")
        
        elif method == "Forward fill":
            data_copy = data_copy.fillna(method='ffill')
            st.info("Applied forward fill to missing values")
        
        elif method == "Backward fill":
            data_copy = data_copy.fillna(method='bfill')
            st.info("Applied backward fill to missing values")
        
        return data_copy
    
    def encode_categorical_variables(self, data: pd.DataFrame, columns: List[str], 
                                   method: str) -> pd.DataFrame:
        """Encode categorical variables using specified method"""
        data_copy = data.copy()
        
        if method == "Label Encoding":
            for col in columns:
                if col in data_copy.columns:
                    le = LabelEncoder()
                    data_copy[col] = le.fit_transform(data_copy[col].astype(str))
                    self.encoders[col] = le
                    st.info(f"Applied Label Encoding to {col}")
        
        elif method == "One-Hot Encoding":
            # Use pandas get_dummies for simplicity
            original_cols = data_copy.shape[1]
            data_copy = pd.get_dummies(data_copy, columns=columns, prefix=columns)
            new_cols = data_copy.shape[1]
            st.info(f"Applied One-Hot Encoding: {original_cols} → {new_cols} columns")
        
        elif method == "Ordinal Encoding":
            for col in columns:
                if col in data_copy.columns:
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    data_copy[col] = oe.fit_transform(data_copy[[col]]).flatten()
                    self.encoders[col] = oe
                    st.info(f"Applied Ordinal Encoding to {col}")
        
        return data_copy
    
    def scale_numerical_features(self, data: pd.DataFrame, columns: List[str], 
                               method: str) -> pd.DataFrame:
        """Scale numerical features using specified method"""
        data_copy = data.copy()
        
        if method == "StandardScaler":
            scaler = StandardScaler()
        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif method == "RobustScaler":
            scaler = RobustScaler()
        elif method == "Normalizer":
            scaler = Normalizer()
        else:
            st.warning(f"Unknown scaling method: {method}")
            return data_copy
        
        # Apply scaling
        if columns:
            data_copy[columns] = scaler.fit_transform(data_copy[columns])
            
            # Store scaler for later use
            for col in columns:
                self.scalers[col] = scaler
            
            st.info(f"Applied {method} to {len(columns)} columns")
        
        return data_copy
    
    def detect_outliers(self, data: pd.DataFrame, columns: List[str] = None, 
                       method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in numerical columns"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(data)) * 100,
                    'indices': data[outlier_mask].index.tolist(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask = z_scores > 3
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(data)) * 100,
                    'indices': data[outlier_mask].index.tolist(),
                    'threshold': 3
                }
        
        return outliers
    
    def handle_outliers(self, data: pd.DataFrame, outliers: Dict[str, Any], 
                       method: str = 'clip') -> pd.DataFrame:
        """Handle outliers using specified method"""
        data_copy = data.copy()
        
        for col, outlier_info in outliers.items():
            if col not in data_copy.columns:
                continue
                
            if method == 'remove':
                # Remove outlier rows
                outlier_indices = outlier_info['indices']
                data_copy = data_copy.drop(outlier_indices)
                st.info(f"Removed {len(outlier_indices)} outliers from {col}")
            
            elif method == 'clip':
                # Clip outliers to bounds
                if 'lower_bound' in outlier_info and 'upper_bound' in outlier_info:
                    data_copy[col] = data_copy[col].clip(
                        lower=outlier_info['lower_bound'],
                        upper=outlier_info['upper_bound']
                    )
                    st.info(f"Clipped outliers in {col} to bounds")
            
            elif method == 'transform':
                # Log transformation for positive values
                if (data_copy[col] > 0).all():
                    data_copy[col] = np.log1p(data_copy[col])
                    st.info(f"Applied log transformation to {col}")
                else:
                    st.warning(f"Cannot apply log transformation to {col} (contains non-positive values)")
        
        return data_copy
    
    def create_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create new features through feature engineering"""
        data_copy = data.copy()
        
        # Numerical feature engineering
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Create interaction features for first two numeric columns
            col1, col2 = numeric_cols[0], numeric_cols[1]
            
            # Interaction term
            data_copy[f'{col1}_{col2}_interaction'] = data_copy[col1] * data_copy[col2]
            
            # Ratio feature
            if (data_copy[col2] != 0).all():
                data_copy[f'{col1}_{col2}_ratio'] = data_copy[col1] / data_copy[col2]
            
            st.info(f"Created interaction and ratio features for {col1} and {col2}")
        
        # Polynomial features
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            data_copy[f'{col}_squared'] = data_copy[col] ** 2
            data_copy[f'{col}_sqrt'] = np.sqrt(np.abs(data_copy[col]))
            st.info(f"Created polynomial features for {col}")
        
        return data_copy
    
    def get_preprocessing_summary(self, original_data: pd.DataFrame, 
                                processed_data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of preprocessing operations"""
        summary = {
            'original_shape': original_data.shape,
            'processed_shape': processed_data.shape,
            'original_missing': original_data.isnull().sum().sum(),
            'processed_missing': processed_data.isnull().sum().sum(),
            'original_dtypes': original_data.dtypes.value_counts().to_dict(),
            'processed_dtypes': processed_data.dtypes.value_counts().to_dict(),
            'columns_added': set(processed_data.columns) - set(original_data.columns),
            'columns_removed': set(original_data.columns) - set(processed_data.columns),
            'encoders_used': list(self.encoders.keys()),
            'scalers_used': list(self.scalers.keys())
        }
        
        return summary
    
    def render_preprocessing_tab(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Render preprocessing interface"""
        st.header("⚙️ Data Preprocessing")
        
        if data is None:
            st.warning("Please upload data first")
            return None
        
        # Create tabs for different preprocessing steps
        tabs = st.tabs([
            "🔧 Basic Cleaning",
            "📊 Encoding",
            "📏 Scaling",
            "🎯 Outliers",
            "🔨 Feature Engineering"
        ])
        
        processed_data = data.copy()
        
        # Basic Cleaning Tab
        with tabs[0]:
            st.subheader("Missing Values Handling")
            
            # Show current missing values
            missing_info = data.isnull().sum()
            missing_info = missing_info[missing_info > 0]
            
            if not missing_info.empty:
                st.write("Columns with missing values:")
                st.dataframe(missing_info.to_frame(name='Missing Count'))
                
                missing_method = st.selectbox(
                    "How to handle missing values?",
                    ["Drop rows with missing values", "Fill with mean (numeric)", 
                     "Fill with median (numeric)", "Fill with mode", "Forward fill", "Backward fill"]
                )
                
                if st.button("Apply Missing Values Handling"):
                    processed_data = self.handle_missing_values(processed_data, missing_method)
                    st.session_state.processed_data = processed_data
                    st.success("Missing values handled successfully!")
            else:
                st.success("No missing values found! 🎉")
        
        # Encoding Tab
        with tabs[1]:
            st.subheader("Categorical Variables Encoding")
            
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols:
                selected_cat_cols = st.multiselect(
                    "Select categorical columns to encode:",
                    categorical_cols,
                    default=categorical_cols
                )
                
                encoding_method = st.selectbox(
                    "Encoding method:",
                    ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"]
                )
                
                if st.button("Apply Encoding") and selected_cat_cols:
                    processed_data = self.encode_categorical_variables(
                        processed_data, selected_cat_cols, encoding_method
                    )
                    st.session_state.processed_data = processed_data
                    st.success("Categorical encoding applied successfully!")
            else:
                st.info("No categorical columns found")
        
        # Scaling Tab
        with tabs[2]:
            st.subheader("Feature Scaling")
            
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_cols:
                selected_num_cols = st.multiselect(
                    "Select numerical columns to scale:",
                    numerical_cols,
                    default=numerical_cols
                )
                
                scaling_method = st.selectbox(
                    "Scaling method:",
                    ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"]
                )
                
                if st.button("Apply Scaling") and selected_num_cols:
                    processed_data = self.scale_numerical_features(
                        processed_data, selected_num_cols, scaling_method
                    )
                    st.session_state.processed_data = processed_data
                    st.success("Feature scaling applied successfully!")
            else:
                st.info("No numerical columns found")
        
        # Outliers Tab
        with tabs[3]:
            st.subheader("Outlier Detection and Handling")
            
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_cols:
                selected_outlier_cols = st.multiselect(
                    "Select columns for outlier detection:",
                    numerical_cols,
                    default=numerical_cols[:3]  # Default to first 3 columns
                )
                
                outlier_method = st.selectbox(
                    "Outlier detection method:",
                    ["iqr", "zscore"]
                )
                
                if st.button("Detect Outliers") and selected_outlier_cols:
                    outliers = self.detect_outliers(processed_data, selected_outlier_cols, outlier_method)
                    st.session_state.outliers = outliers
                    
                    # Display outlier information
                    for col, info in outliers.items():
                        if info['count'] > 0:
                            st.warning(f"**{col}**: {info['count']} outliers ({info['percentage']:.1f}%)")
                        else:
                            st.success(f"**{col}**: No outliers detected")
                
                # Handle outliers if detected
                if 'outliers' in st.session_state:
                    outlier_handling = st.selectbox(
                        "How to handle outliers?",
                        ["keep", "remove", "clip", "transform"]
                    )
                    
                    if st.button("Apply Outlier Handling") and outlier_handling != "keep":
                        processed_data = self.handle_outliers(
                            processed_data, st.session_state.outliers, outlier_handling
                        )
                        st.session_state.processed_data = processed_data
                        st.success("Outlier handling applied successfully!")
        
        # Feature Engineering Tab
        with tabs[4]:
            st.subheader("Feature Engineering")
            
            if st.button("Create Engineered Features"):
                processed_data = self.create_feature_engineering(processed_data)
                st.session_state.processed_data = processed_data
                st.success("Feature engineering applied successfully!")
        
        # Show preprocessing summary
        if not processed_data.equals(data):
            st.subheader("📋 Preprocessing Summary")
            summary = self.get_preprocessing_summary(data, processed_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Shape", f"{summary['original_shape'][0]} × {summary['original_shape'][1]}")
                st.metric("Original Missing", summary['original_missing'])
            
            with col2:
                st.metric("Processed Shape", f"{summary['processed_shape'][0]} × {summary['processed_shape'][1]}")
                st.metric("Processed Missing", summary['processed_missing'])
            
            if summary['columns_added']:
                st.success(f"Added columns: {', '.join(summary['columns_added'])}")
            
            if summary['columns_removed']:
                st.warning(f"Removed columns: {', '.join(summary['columns_removed'])}")
        
        return processed_data
