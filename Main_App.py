"""
MatrixLab AI Studio - Main Application
Complete ML workflow with all modules integrated
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all modules
from modules.data_loader import DataLoader
from modules.eda import EDAGenerator
from modules.preprocessor import DataPreprocessor
from modules.trainer import ModelTrainer
from modules.predictor import ModelPredictor
from modules.evaluator import ModelEvaluator
from modules.explainability import ModelExplainer
from modules.manager import ModelManager
from modules.visualization import Visualizer
from utils.session_utils import SessionManager

# Page configuration
st.set_page_config(
    page_title="MatrixLab AI Studio",
    page_icon="📊",
    layout="wide"
)

# Initialize session manager
session_manager = SessionManager()

# Initialize modules
data_loader = DataLoader()
eda_generator = EDAGenerator()
preprocessor = DataPreprocessor()
trainer = ModelTrainer()
predictor = ModelPredictor()
evaluator = ModelEvaluator()
explainer = ModelExplainer()
manager = ModelManager()
visualizer = Visualizer()

# App header
st.title("📊 MatrixLab AI Studio")
st.markdown("### Complete Machine Learning Workflow")

# Sidebar navigation
st.sidebar.markdown("## 🧭 Workflow Navigation")

# Workflow steps
workflow_steps = [
    "📁 Data Upload & Visualization",
    "🔍 Data Exploration", 
    "⚙️ Data Preprocessing",
    "🎯 Model Training",
    "📈 Model Evaluation",
    "🔮 Predictions",
    "🧠 Model Explainability",
    "💾 Model Management"
]

# Progress tracking
current_step = st.session_state.get('current_step', 0)
st.sidebar.markdown(f"**Current Step**: {current_step + 1}/8")

# Step selection
selected_step = st.sidebar.selectbox(
    "Select Workflow Step",
    range(len(workflow_steps)),
    format_func=lambda x: workflow_steps[x],
    index=current_step
)

# Update current step
if selected_step != current_step:
    st.session_state.current_step = selected_step
    st.rerun()

# Progress bar
progress = (selected_step + 1) / len(workflow_steps)
st.sidebar.progress(progress)

# Session info
st.sidebar.markdown("---")
st.sidebar.markdown("## 📋 Session Info")
st.sidebar.markdown(f"**Project**: {st.session_state.get('project_name', 'Untitled')}")
st.sidebar.markdown(f"**Data Status**: {'✅ Loaded' if st.session_state.get('data_uploaded', False) else '❌ Not Loaded'}")

# Main content area
st.markdown("---")

# Step 1: Data Upload
if selected_step == 0:
    st.header("📁 Step 1: Data Upload & Visualization")
    
    # Project name input
    project_name = st.text_input("Project Name", value=st.session_state.get('project_name', 'New Project'))
    if project_name != st.session_state.get('project_name', 'New Project'):
        st.session_state.project_name = project_name
    
    # Data upload
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'xlsx', 'json'],
        help="Supported formats: CSV, Excel, JSON"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = data_loader.load_file(uploaded_file)
            
            if df is not None:
                st.session_state.data = df
                st.session_state.data_uploaded = True
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Display basic info
                st.subheader("📊 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", f"{df.shape[1]:,}")
                with col3:
                    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                
                # Data preview
                st.subheader("🔍 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data types
                st.subheader("📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns.tolist(),
                    'Data Type': df.dtypes.astype(str).tolist(),
                    'Missing': df.isnull().sum().astype(int).tolist(),
                    'Unique Values': df.nunique().astype(int).tolist()
                })
                st.dataframe(col_info, use_container_width=True)
                
                # === DATA VISUALIZATION SECTION ===
                st.markdown("---")
                st.subheader("📊 Data Visualization & Analysis")
                st.markdown("*Explore your data through interactive charts and graphs*")
                
                # Create tabs for different visualization categories
                viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
                    "📈 Distribution Plots", 
                    "🔗 Relationships", 
                    "📊 Categories", 
                    "🎯 Missing Data",
                    "📋 Statistical Overview"
                ])
                
                # Get numeric and categorical columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                with viz_tab1:
                    st.markdown("### 📈 Distribution Analysis")
                    
                    if numeric_cols:
                        # Distribution plots for numeric columns
                        st.markdown("#### Numerical Distributions")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_num_col = st.selectbox("Select Column", numeric_cols, key="dist_col")
                        with col2:
                            plot_type = st.selectbox("Plot Type", ["histogram", "box", "violin"], key="dist_type")
                        
                        if selected_num_col:
                            fig = visualizer.create_distribution_plot(df, selected_num_col, plot_type)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Multi-column distribution grid
                        st.markdown("#### Feature Distribution Grid")
                        selected_dist_cols = st.multiselect(
                            "Select columns for distribution grid", 
                            numeric_cols, 
                            default=numeric_cols[:min(6, len(numeric_cols))],
                            key="dist_grid_cols"
                        )
                        
                        if selected_dist_cols:
                            fig_grid = visualizer.create_feature_distribution_grid(df, selected_dist_cols)
                            if fig_grid:
                                st.plotly_chart(fig_grid, use_container_width=True)
                    else:
                        st.info("No numerical columns found for distribution analysis")
                
                with viz_tab2:
                    st.markdown("### 🔗 Relationship Analysis")
                    
                    if len(numeric_cols) >= 2:
                        # Correlation heatmap
                        st.markdown("#### Correlation Matrix")
                        fig_corr = visualizer.create_correlation_heatmap(df)
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Scatter plots
                        st.markdown("#### Scatter Plot Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                        with col2:
                            y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col], key="scatter_y")
                        with col3:
                            color_col = st.selectbox("Color by (optional)", 
                                                   [None] + categorical_cols + numeric_cols, 
                                                   key="scatter_color")
                        
                        if x_col and y_col:
                            fig_scatter = visualizer.create_scatter_plot(df, x_col, y_col, color_col)
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Pairplot for selected columns
                        if len(numeric_cols) >= 2:
                            st.markdown("#### Pairplot Matrix")
                            selected_pair_cols = st.multiselect(
                                "Select columns for pairplot", 
                                numeric_cols, 
                                default=numeric_cols[:min(4, len(numeric_cols))],
                                key="pair_cols"
                            )
                            
                            if len(selected_pair_cols) >= 2:
                                fig_pair = visualizer.create_pairplot_matrix(df, selected_pair_cols)
                                if fig_pair:
                                    st.plotly_chart(fig_pair, use_container_width=True)
                    else:
                        st.info("Need at least 2 numerical columns for relationship analysis")
                
                with viz_tab3:
                    st.markdown("### 📊 Categorical Analysis")
                    
                    if categorical_cols:
                        # Categorical distribution plots
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            selected_cat_col = st.selectbox("Select Column", categorical_cols, key="cat_col")
                        with col2:
                            cat_plot_type = st.selectbox("Plot Type", ["bar", "pie"], key="cat_plot_type")
                        
                        if selected_cat_col:
                            fig_cat = visualizer.create_categorical_plot(df, selected_cat_col, cat_plot_type)
                            if fig_cat:
                                st.plotly_chart(fig_cat, use_container_width=True)
                        
                        # Value counts table
                        st.markdown("#### Value Counts")
                        if selected_cat_col:
                            value_counts = df[selected_cat_col].value_counts().reset_index()
                            value_counts.columns = [selected_cat_col, 'Count']
                            value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(2)
                            st.dataframe(value_counts, use_container_width=True)
                        
                        # Cross-tabulation if multiple categorical columns
                        if len(categorical_cols) >= 2:
                            st.markdown("#### Cross-Tabulation")
                            col1, col2 = st.columns(2)
                            with col1:
                                cross_col1 = st.selectbox("First Column", categorical_cols, key="cross_col1")
                            with col2:
                                cross_col2 = st.selectbox("Second Column", 
                                                        [col for col in categorical_cols if col != cross_col1], 
                                                        key="cross_col2")
                            
                            if cross_col1 and cross_col2:
                                crosstab = pd.crosstab(df[cross_col1], df[cross_col2])
                                st.dataframe(crosstab, use_container_width=True)
                    else:
                        st.info("No categorical columns found for analysis")
                
                with viz_tab4:
                    st.markdown("### 🎯 Missing Data Analysis")
                    
                    # Missing values visualization
                    fig_missing = visualizer.create_missing_values_plot(df)
                    st.plotly_chart(fig_missing, use_container_width=True)
                    
                    # Missing data patterns
                    missing_summary = df.isnull().sum()
                    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
                    
                    if not missing_summary.empty:
                        st.markdown("#### Missing Data Summary")
                        missing_df = pd.DataFrame({
                            'Column': missing_summary.index,
                            'Missing Count': missing_summary.values,
                            'Missing Percentage': (missing_summary.values / len(df) * 100).round(2)
                        })
                        st.dataframe(missing_df, use_container_width=True)
                        
                        # Missing data heatmap for patterns
                        if len(missing_summary) > 1:
                            st.markdown("#### Missing Data Pattern")
                            missing_pattern = df[missing_summary.index].isnull().astype(int)
                            if missing_pattern.shape[1] > 1:
                                fig_pattern = visualizer.create_correlation_heatmap(
                                    missing_pattern, 
                                    "Missing Data Pattern Correlation"
                                )
                                if fig_pattern:
                                    st.plotly_chart(fig_pattern, use_container_width=True)
                    else:
                        st.success("🎉 No missing values found in your dataset!")
                
                with viz_tab5:
                    st.markdown("### 📋 Statistical Overview")
                    
                    # Statistical summary
                    if numeric_cols:
                        st.markdown("#### Numerical Statistics")
                        stats_df = df[numeric_cols].describe().round(3)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Outlier detection
                        st.markdown("#### Outlier Detection")
                        selected_outlier_cols = st.multiselect(
                            "Select columns for outlier analysis", 
                            numeric_cols,
                            default=numeric_cols[:min(5, len(numeric_cols))],
                            key="outlier_cols"
                        )
                        
                        if selected_outlier_cols:
                            fig_outliers = visualizer.create_outlier_detection_plot(df, selected_outlier_cols)
                            if fig_outliers:
                                st.plotly_chart(fig_outliers, use_container_width=True)
                    
                    # Data quality metrics
                    st.markdown("#### Data Quality Metrics")
                    quality_metrics = {
                        'Total Rows': len(df),
                        'Total Columns': len(df.columns),
                        'Numerical Columns': len(numeric_cols),
                        'Categorical Columns': len(categorical_cols),
                        'Missing Values': df.isnull().sum().sum(),
                        'Duplicate Rows': df.duplicated().sum(),
                        'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
                    }
                    
                    # Display metrics in columns
                    metric_cols = st.columns(len(quality_metrics))
                    for i, (metric, value) in enumerate(quality_metrics.items()):
                        with metric_cols[i]:
                            st.metric(metric, value)
                    
                    # Column-wise summary
                    st.markdown("#### Column Summary")
                    col_summary = pd.DataFrame({
                        'Column': df.columns.tolist(),
                        'Data Type': df.dtypes.astype(str).tolist(),
                        'Non-Null Count': df.count().astype(int).tolist(),
                        'Null Count': df.isnull().sum().astype(int).tolist(),
                        'Unique Values': df.nunique().astype(int).tolist(),
                        'Memory Usage (KB)': (df.memory_usage(deep=True) / 1024).round(2).astype(float).tolist()
                    })
                    st.dataframe(col_summary, use_container_width=True)
                
                st.markdown("---")
                
                # Next step button
                if st.button("➡️ Proceed to Data Exploration", type="primary"):
                    st.session_state.current_step = 1
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.info("👆 Please upload a dataset to begin")

# Step 2: Data Exploration
elif selected_step == 1:
    st.header("🔍 Step 2: Data Exploration")
    
    if not st.session_state.get('data_uploaded', False):
        st.warning("⚠️ Please upload data first")
        if st.button("⬅️ Go to Data Upload"):
            st.session_state.current_step = 0
            st.rerun()
    else:
        df = st.session_state.data
        
        # EDA options
        st.subheader("📊 Exploratory Data Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Generate Full EDA Report", type="primary"):
                with st.spinner("Generating comprehensive EDA report..."):
                    report = eda_generator.generate_profile_report(df)
                    st.session_state.eda_report = report
                    st.success("✅ EDA report generated!")
        
        with col2:
            if st.button("📈 Quick Statistical Summary"):
                summary = eda_generator.generate_quick_summary(df)
                st.session_state.eda_summary = summary
                st.success("✅ Statistical summary generated!")
        
        # Display EDA results
        if 'eda_report' in st.session_state:
            st.subheader("📋 Complete EDA Report")
            st.components.v1.html(st.session_state.eda_report, height=800, scrolling=True)
        
        if 'eda_summary' in st.session_state:
            st.subheader("📈 Statistical Summary")
            summary = st.session_state.eda_summary
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Numerical Columns", len(summary['numerical_columns']))
            with col2:
                st.metric("Categorical Columns", len(summary['categorical_columns']))
            with col3:
                st.metric("Missing Values", summary['total_missing'])
            with col4:
                st.metric("Duplicate Rows", summary['duplicates'])
            
            # Visualizations
            if summary['correlation_matrix'] is not None:
                st.subheader("🔗 Correlation Matrix")
                fig = eda_generator.create_correlation_heatmap(summary['correlation_matrix'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Data Upload"):
                st.session_state.current_step = 0
                st.rerun()
        with col2:
            if st.button("➡️ Proceed to Preprocessing", type="primary"):
                st.session_state.current_step = 2
                st.rerun()

# Step 3: Data Preprocessing
elif selected_step == 2:
    st.header("⚙️ Step 3: Data Preprocessing")
    
    if not st.session_state.get('data_uploaded', False):
        st.warning("⚠️ Please upload data first")
        if st.button("⬅️ Go to Data Upload"):
            st.session_state.current_step = 0
            st.rerun()
    else:
        df = st.session_state.data
        
        # Preprocessing options
        st.subheader("🔧 Preprocessing Options")
        
        # Missing values handling
        st.markdown("#### Missing Values")
        missing_method = st.selectbox(
            "How to handle missing values?",
            ["Drop rows with missing values", "Fill with mean (numeric)", "Fill with median (numeric)", 
             "Fill with mode", "Forward fill", "Backward fill"]
        )
        
        # Feature selection
        st.markdown("#### Feature Selection")
        all_columns = df.columns.tolist()
        selected_features = st.multiselect(
            "Select features to include",
            all_columns,
            default=all_columns
        )
        
        # Target column
        target_column = st.selectbox(
            "Select target column",
            all_columns,
            index=len(all_columns)-1 if all_columns else 0
        )
        
        # Preprocessing execution
        if st.button("🔄 Apply Preprocessing", type="primary"):
            try:
                # Apply preprocessing
                processed_df = preprocessor.handle_missing_values(df, missing_method)
                
                # Store preprocessing info
                st.session_state.processed_data = processed_df
                st.session_state.selected_features = selected_features
                st.session_state.target_column = target_column
                st.session_state.preprocessing_applied = True
                
                st.success("✅ Preprocessing completed!")
                
                # Show results
                st.subheader("📊 Preprocessing Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
                with col2:
                    st.metric("Processed Shape", f"{processed_df.shape[0]} × {processed_df.shape[1]}")
                with col3:
                    st.metric("Missing Values", f"{processed_df.isnull().sum().sum()}")
                
                # Preview processed data
                st.subheader("🔍 Processed Data Preview")
                st.dataframe(processed_df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in preprocessing: {str(e)}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Data Exploration"):
                st.session_state.current_step = 1
                st.rerun()
        with col2:
            if st.button("➡️ Proceed to Model Training", type="primary"):
                if st.session_state.get('preprocessing_applied', False):
                    st.session_state.current_step = 3
                    st.rerun()
                else:
                    st.warning("Please apply preprocessing first")

# Step 4: Model Training
elif selected_step == 3:
    st.header("🎯 Step 4: Model Training")
    
    if not st.session_state.get('preprocessing_applied', False):
        st.warning("⚠️ Please complete preprocessing first")
        if st.button("⬅️ Go to Preprocessing"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        # Training interface
        trainer.render_training_tab(st.session_state.processed_data)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Preprocessing"):
                st.session_state.current_step = 2
                st.rerun()
        with col2:
            if st.button("➡️ Proceed to Evaluation", type="primary"):
                st.session_state.current_step = 4
                st.rerun()

# Step 5: Model Evaluation
elif selected_step == 4:
    st.header("📈 Step 5: Model Evaluation")
    
    if not st.session_state.get('training_results'):
        st.warning("⚠️ Please train models first")
        if st.button("⬅️ Go to Model Training"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        # Evaluation interface
        evaluator.render_evaluation_tab(st.session_state.training_results)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Training"):
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button("➡️ Proceed to Predictions", type="primary"):
                st.session_state.current_step = 5
                st.rerun()

# Step 6: Predictions
elif selected_step == 5:
    st.header("🔮 Step 6: Predictions")
    
    # Prediction interface
    predictor.render_prediction_tab(st.session_state.get('training_results'))
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Evaluation"):
            st.session_state.current_step = 4
            st.rerun()
    with col2:
        if st.button("➡️ Proceed to Explainability", type="primary"):
            st.session_state.current_step = 6
            st.rerun()

# Step 7: Model Explainability
elif selected_step == 6:
    st.header("🧠 Step 7: Model Explainability")
    
    # Explainability interface
    explainer.render_explainability_tab(
        st.session_state.get('training_results'),
        st.session_state.get('processed_data')
    )
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Predictions"):
            st.session_state.current_step = 5
            st.rerun()
    with col2:
        if st.button("➡️ Proceed to Model Management", type="primary"):
            st.session_state.current_step = 7
            st.rerun()

# Step 8: Model Management
elif selected_step == 7:
    st.header("💾 Step 8: Model Management")
    
    # Model management interface
    manager.render_management_tab(st.session_state.get('training_results'))
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Explainability"):
            st.session_state.current_step = 6
            st.rerun()
    with col2:
        if st.button("🔄 Start New Project", type="primary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key not in ['current_step', 'project_name']:
                    del st.session_state[key]
            st.session_state.current_step = 0
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**MatrixLab AI Studio** | Complete ML Workflow Platform")

#.\venv\Scripts\activate
#streamlit run Main_App.py
#streamlit run MachineLearningStudio/Main_App.py