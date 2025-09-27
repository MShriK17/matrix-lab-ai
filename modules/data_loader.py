"""
Data Loading Module for MatrixLab AI Studio
Handles various data formats and file uploads
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Optional, Union
import io

class DataLoader:
    """Handles data loading from various sources and formats"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls', 'json']
    
    def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return self._load_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return self._load_excel(uploaded_file)
            elif file_extension == 'json':
                return self._load_json(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def _load_csv(self, uploaded_file) -> pd.DataFrame:
        """Load CSV file with encoding detection"""
        try:
            # Try common encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"✅ CSV loaded successfully with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
            st.warning("⚠️ Some characters might be incorrectly decoded")
            return df
            
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return None
    
    def _load_excel(self, uploaded_file) -> pd.DataFrame:
        """Load Excel file"""
        try:
            # Check if file has multiple sheets
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Select sheet to load:",
                    sheet_names,
                    key="excel_sheet_selector"
                )
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Excel file loaded successfully")
            return df
            
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None
    
    def _load_json(self, uploaded_file) -> pd.DataFrame:
        """Load JSON file"""
        try:
            json_data = json.load(uploaded_file)
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # Try to normalize nested JSON
                df = pd.json_normalize(json_data)
            else:
                st.error("Unsupported JSON structure")
                return None
            
            st.success(f"✅ JSON file loaded successfully")
            return df
            
        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """Validate loaded data and return summary"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'summary': {}
        }
        
        try:
            # Basic validation
            if df.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Dataset is empty")
                return validation_results
            
            # Check for minimum requirements
            if df.shape[0] < 2:
                validation_results['warnings'].append("Dataset has very few rows")
            
            if df.shape[1] < 2:
                validation_results['warnings'].append("Dataset has very few columns")
            
            # Check for excessive missing values
            missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_percentage > 50:
                validation_results['warnings'].append(f"High percentage of missing values: {missing_percentage:.1f}%")
            
            # Summary statistics
            validation_results['summary'] = {
                'shape': df.shape,
                'dtypes': df.dtypes.value_counts().to_dict(),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': missing_percentage,
                'duplicate_rows': df.duplicated().sum(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def preview_data(self, df: pd.DataFrame, n_rows: int = 5) -> None:
        """Display data preview with basic statistics"""
        st.subheader("📊 Data Preview")
        
        # Display sample data
        st.dataframe(df.head(n_rows), use_container_width=True)
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("Duplicates", f"{df.duplicated().sum():,}")
        
        # Column information
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
