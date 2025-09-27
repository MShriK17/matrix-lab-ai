"""
Model Explainability Module for MatrixLab AI Studio
SHAP and LIME implementations for model interpretability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """Model explainability using SHAP and LIME"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.shap_available = False
        self.lime_available = False
        
        # Check for SHAP availability
        try:
            import shap
            self.shap_available = True
            self.shap = shap
        except ImportError:
            self.shap_available = False
        
        # Check for LIME availability
        try:
            import lime
            import lime.lime_tabular
            self.lime_available = True
            self.lime = lime
        except ImportError:
            self.lime_available = False
    
    def create_feature_importance_plot(self, importance_values: np.ndarray, 
                                     feature_names: List[str],
                                     title: str = "Feature Importance") -> go.Figure:
        """Create feature importance visualization"""
        if len(importance_values) != len(feature_names):
            st.error("Length mismatch between importance values and feature names")
            return None
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(importance_values)
        }).sort_values('importance', ascending=True)
        
        # Create color based on positive/negative impact
        colors = ['red' if val < 0 else 'blue' for val in importance_values[importance_df.index]]
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:.3f}" for val in importance_df['importance']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Features',
            height=max(400, len(feature_names) * 25),
            margin=dict(l=150)
        )
        
        return fig
    
    def explain_with_shap(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                         feature_names: List[str], max_samples: int = 100) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        if not self.shap_available:
            st.error("SHAP is not available. Please install it to use SHAP explanations.")
            return None
        
        try:
            # Limit samples for performance
            X_test_sample = X_test[:max_samples] if len(X_test) > max_samples else X_test
            X_train_sample = X_train[:min(100, len(X_train))]
            
            # Initialize explainer based on model type
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                if hasattr(model, 'tree_'):
                    explainer = self.shap.TreeExplainer(model)
                else:
                    # For other models, use KernelExplainer with sample background
                    explainer = self.shap.KernelExplainer(model.predict_proba, X_train_sample)
            else:
                # Regression models
                if hasattr(model, 'tree_'):
                    explainer = self.shap.TreeExplainer(model)
                else:
                    explainer = self.shap.KernelExplainer(model.predict, X_train_sample)
            
            # Calculate SHAP values
            with st.spinner("Calculating SHAP values..."):
                shap_values = explainer.shap_values(X_test_sample)
            
            return {
                'explainer': explainer,
                'shap_values': shap_values,
                'X_test_sample': X_test_sample,
                'feature_names': feature_names
            }
            
        except Exception as e:
            st.error(f"Error generating SHAP explanations: {str(e)}")
            return None
    
    def create_shap_summary_plot(self, shap_values: np.ndarray, X_test: np.ndarray, 
                                feature_names: List[str]) -> go.Figure:
        """Create SHAP summary plot using Plotly"""
        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class classification - use first class
                shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
            else:
                shap_vals = shap_values
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap = np.abs(shap_vals).mean(axis=0)
            
            # Create DataFrame for plotting
            summary_df = pd.DataFrame({
                'feature': feature_names,
                'mean_shap': mean_shap
            }).sort_values('mean_shap', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=summary_df['mean_shap'],
                y=summary_df['feature'],
                orientation='h',
                marker_color=self.color_palette[0],
                text=[f"{val:.3f}" for val in summary_df['mean_shap']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance Summary",
                xaxis_title='Mean |SHAP Value|',
                yaxis_title='Features',
                height=max(400, len(feature_names) * 25),
                margin=dict(l=150)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating SHAP summary plot: {str(e)}")
            return None
    
    def create_shap_waterfall_plot(self, shap_values: np.ndarray, X_test: np.ndarray,
                                  feature_names: List[str], instance_idx: int = 0) -> go.Figure:
        """Create SHAP waterfall plot for a single prediction"""
        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_vals = shap_values[0][instance_idx] if len(shap_values) > 0 else shap_values[instance_idx]
            else:
                shap_vals = shap_values[instance_idx]
            
            # Create waterfall data
            base_value = 0  # Simplified base value
            cumulative = base_value
            
            # Sort by absolute SHAP value
            indices = np.argsort(np.abs(shap_vals))[::-1]
            
            waterfall_data = []
            for i, idx in enumerate(indices):
                waterfall_data.append({
                    'feature': feature_names[idx],
                    'shap_value': shap_vals[idx],
                    'cumulative': cumulative + shap_vals[idx],
                    'feature_value': X_test[instance_idx, idx]
                })
                cumulative += shap_vals[idx]
            
            # Create plot
            fig = go.Figure()
            
            x_pos = list(range(len(waterfall_data)))
            colors = ['red' if val['shap_value'] < 0 else 'blue' for val in waterfall_data]
            
            fig.add_trace(go.Bar(
                x=x_pos,
                y=[val['shap_value'] for val in waterfall_data],
                marker_color=colors,
                text=[f"{val['feature']}<br>SHAP: {val['shap_value']:.3f}<br>Value: {val['feature_value']:.3f}" 
                      for val in waterfall_data],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"SHAP Waterfall Plot - Instance {instance_idx}",
                xaxis_title='Features',
                yaxis_title='SHAP Value',
                xaxis=dict(tickvals=x_pos, ticktext=[val['feature'] for val in waterfall_data]),
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating SHAP waterfall plot: {str(e)}")
            return None
    
    def explain_with_lime(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                         feature_names: List[str], instance_idx: int = 0,
                         problem_type: str = "classification") -> Optional[Dict]:
        """Generate LIME explanation for a single instance"""
        if not self.lime_available:
            st.error("LIME is not available. Please install it to use LIME explanations.")
            return None
        
        try:
            # Create LIME explainer
            if problem_type == "classification":
                explainer = self.lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=['Class 0', 'Class 1'] if hasattr(model, 'predict_proba') else None,
                    mode='classification'
                )
                
                explanation = explainer.explain_instance(
                    X_test[instance_idx],
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=len(feature_names)
                )
            else:
                explainer = self.lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    mode='regression'
                )
                
                explanation = explainer.explain_instance(
                    X_test[instance_idx],
                    model.predict,
                    num_features=len(feature_names)
                )
            
            return {
                'explainer': explainer,
                'explanation': explanation,
                'instance_idx': instance_idx
            }
            
        except Exception as e:
            st.error(f"Error generating LIME explanation: {str(e)}")
            return None
    
    def create_lime_plot(self, lime_explanation) -> go.Figure:
        """Create LIME explanation plot"""
        try:
            # Extract feature importance from LIME explanation
            feature_importance = lime_explanation['explanation'].as_list()
            
            features = [item[0] for item in feature_importance]
            importance = [item[1] for item in feature_importance]
            
            # Create plot
            colors = ['red' if val < 0 else 'blue' for val in importance]
            
            fig = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker_color=colors,
                text=[f"{val:.3f}" for val in importance],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"LIME Explanation - Instance {lime_explanation['instance_idx']}",
                xaxis_title='Feature Importance',
                yaxis_title='Features',
                height=max(400, len(features) * 25),
                margin=dict(l=150)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating LIME plot: {str(e)}")
            return None
    
    def create_partial_dependence_plot(self, model: Any, X: np.ndarray, 
                                     feature_names: List[str], 
                                     feature_idx: int) -> go.Figure:
        """Create partial dependence plot for a feature"""
        try:
            feature_name = feature_names[feature_idx]
            
            # Create range of values for the feature
            feature_values = X[:, feature_idx]
            feature_range = np.linspace(feature_values.min(), feature_values.max(), 50)
            
            # Calculate partial dependence
            partial_dependence = []
            
            for value in feature_range:
                # Create modified dataset with feature set to specific value
                X_modified = X.copy()
                X_modified[:, feature_idx] = value
                
                # Predict and average
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(X_modified)[:, 1]  # Use positive class probability
                else:
                    predictions = model.predict(X_modified)
                
                partial_dependence.append(predictions.mean())
            
            # Create plot
            fig = go.Figure(go.Scatter(
                x=feature_range,
                y=partial_dependence,
                mode='lines+markers',
                line=dict(color=self.color_palette[0], width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=f"Partial Dependence Plot - {feature_name}",
                xaxis_title=feature_name,
                yaxis_title='Partial Dependence',
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating partial dependence plot: {str(e)}")
            return None
    
    def render_explainability_tab(self, training_results: Optional[Dict] = None,
                                 data: Optional[pd.DataFrame] = None) -> None:
        """Render comprehensive explainability interface"""
        st.header("🧠 Model Explainability")
        
        if not training_results:
            st.warning("⚠️ No training results found. Please train models first.")
            return
        
        # Check library availability
        if not self.shap_available or not self.lime_available:
            st.info("📋 Advanced explainability libraries not available. Showing built-in feature importance analysis.")
            missing_libs = []
            if not self.shap_available:
                missing_libs.append("SHAP")
            if not self.lime_available:
                missing_libs.append("LIME")
            
            with st.expander("Install Advanced Libraries (Optional)"):
                st.markdown(f"**Missing libraries**: {', '.join(missing_libs)}")
                st.code("pip install shap lime", language="bash")
                st.markdown("These libraries provide advanced model explanations like SHAP values and local interpretability.")
        
        # Model selection
        st.subheader("🎯 Select Model for Explanation")
        model_names = list(training_results.keys())
        selected_model = st.selectbox("Choose model to explain:", model_names)
        
        if not selected_model:
            return
        
        result = training_results[selected_model]
        model = result['model']
        problem_type = result['problem_type']
        X_train = result['X_train']
        X_test = result['X_test']
        
        # Get feature names
        preprocessing_info = st.session_state.get('preprocessing_info', {})
        feature_names = preprocessing_info.get('feature_names', [f'Feature_{i}' for i in range(X_train.shape[1])])
        
        # Create explanation tabs
        if self.shap_available and self.lime_available:
            tab1, tab2, tab3, tab4 = st.tabs(["🔍 Feature Importance", "📊 SHAP Analysis", "🎯 LIME Explanation", "📈 Partial Dependence"])
        else:
            tab1, tab2 = st.tabs(["🔍 Feature Importance", "📈 Partial Dependence"])
            
        with tab1:
            st.markdown("### Built-in Feature Importance")
            
            # Model-specific feature importance
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### Tree-based Model Feature Importance")
                fig = self.create_feature_importance_plot(
                    model.feature_importances_, 
                    feature_names, 
                    f"{selected_model} - Feature Importance"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Show numerical values
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(importance_df, use_container_width=True)
                
            elif hasattr(model, 'coef_'):
                st.markdown("#### Linear Model Coefficients")
                coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                fig = self.create_feature_importance_plot(
                    coefs, 
                    feature_names, 
                    f"{selected_model} - Coefficients"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Show numerical values
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefs,
                    'Abs_Coefficient': np.abs(coefs)
                }).sort_values('Abs_Coefficient', ascending=False)
                st.dataframe(coef_df, use_container_width=True)
            else:
                st.info("This model type doesn't provide built-in feature importance.")
                st.markdown("**Available analysis:**")
                st.markdown("- Partial dependence plots (see next tab)")
                st.markdown("- Model performance metrics (see Evaluation step)")
                
        with tab2:
            st.markdown("### Partial Dependence Analysis")
            st.markdown("Shows how each feature affects predictions on average.")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_feature = st.selectbox(
                    "Select feature:", 
                    range(len(feature_names)),
                    format_func=lambda x: feature_names[x]
                )
            
            if selected_feature is not None:
                fig = self.create_partial_dependence_plot(
                    model, X_train, feature_names, selected_feature
                )
                if fig:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)
                        
        # Advanced tabs only if libraries are available
        if self.shap_available and self.lime_available:
            with tab2:  # SHAP tab
                st.markdown("### SHAP Analysis")
                # SHAP implementation here
                st.info("SHAP analysis would be implemented here with full library")
                
            with tab3:  # LIME tab
                st.markdown("### LIME Explanation")
                # LIME implementation here
                st.info("LIME analysis would be implemented here with full library")
        explanation_tabs = st.tabs(["🔍 SHAP Analysis", "🎯 LIME Analysis", "📊 Partial Dependence", "🏗️ Model-Based Importance"])
        
        # SHAP Analysis Tab
        with explanation_tabs[0]:
            st.subheader("🔍 SHAP (SHapley Additive exPlanations)")
            
            if self.shap_available:
                if st.button("🚀 Generate SHAP Explanations", type="primary"):
                    shap_results = self.explain_with_shap(model, X_train, X_test, feature_names)
                    
                    if shap_results:
                        st.session_state.shap_results = shap_results
                        st.success("✅ SHAP explanations generated successfully!")
                
                # Display SHAP results if available
                if 'shap_results' in st.session_state:
                    shap_results = st.session_state.shap_results
                    
                    # SHAP Summary Plot
                    st.subheader("📊 SHAP Summary Plot")
                    fig_summary = self.create_shap_summary_plot(
                        shap_results['shap_values'],
                        shap_results['X_test_sample'],
                        feature_names
                    )
                    if fig_summary:
                        st.plotly_chart(fig_summary, use_container_width=True)
                    
                    # SHAP Waterfall Plot for individual instances
                    st.subheader("💧 SHAP Waterfall Plot")
                    max_instances = len(shap_results['X_test_sample'])
                    instance_idx = st.slider("Select instance to explain:", 0, max_instances-1, 0)
                    
                    fig_waterfall = self.create_shap_waterfall_plot(
                        shap_results['shap_values'],
                        shap_results['X_test_sample'],
                        feature_names,
                        instance_idx
                    )
                    if fig_waterfall:
                        st.plotly_chart(fig_waterfall, use_container_width=True)
            else:
                st.info("SHAP is not available. Install with: `pip install shap`")
        
        # LIME Analysis Tab
        with explanation_tabs[1]:
            st.subheader("🎯 LIME (Local Interpretable Model-agnostic Explanations)")
            
            if self.lime_available:
                max_instances = len(X_test)
                instance_idx = st.slider("Select instance for LIME explanation:", 0, max_instances-1, 0, key="lime_instance")
                
                if st.button("🔍 Generate LIME Explanation", type="primary"):
                    lime_results = self.explain_with_lime(
                        model, X_train, X_test, feature_names, instance_idx, problem_type.lower()
                    )
                    
                    if lime_results:
                        st.session_state.lime_results = lime_results
                        st.success("✅ LIME explanation generated successfully!")
                
                # Display LIME results if available
                if 'lime_results' in st.session_state:
                    lime_results = st.session_state.lime_results
                    
                    fig_lime = self.create_lime_plot(lime_results)
                    if fig_lime:
                        st.plotly_chart(fig_lime, use_container_width=True)
                    
                    # Show instance details
                    st.subheader("📋 Instance Details")
                    instance_data = X_test[lime_results['instance_idx']]
                    instance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': instance_data
                    })
                    st.dataframe(instance_df, use_container_width=True)
            else:
                st.info("LIME is not available. Install with: `pip install lime`")
        
        # Partial Dependence Tab
        with explanation_tabs[2]:
            st.subheader("📊 Partial Dependence Plots")
            st.markdown("Shows how a feature affects predictions on average.")
            
            selected_feature = st.selectbox(
                "Select feature for partial dependence plot:",
                feature_names,
                key="pd_feature"
            )
            
            if st.button("📈 Generate Partial Dependence Plot"):
                feature_idx = feature_names.index(selected_feature)
                fig_pd = self.create_partial_dependence_plot(model, X_test, feature_names, feature_idx)
                if fig_pd:
                    st.plotly_chart(fig_pd, use_container_width=True)
        
        # Model-Based Importance Tab
        with explanation_tabs[3]:
            st.subheader("🏗️ Model-Based Feature Importance")
            
            if hasattr(model, 'feature_importances_'):
                st.success("✅ Model provides built-in feature importance")
                
                fig_importance = self.create_feature_importance_plot(
                    model.feature_importances_,
                    feature_names,
                    f"Feature Importance - {selected_model}"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Feature importance table
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.subheader("📋 Feature Importance Table")
                st.dataframe(importance_df, use_container_width=True)
                
            elif hasattr(model, 'coef_'):
                st.success("✅ Model provides coefficients")
                
                # For linear models, use coefficients
                if len(model.coef_.shape) == 1:
                    coefficients = model.coef_
                else:
                    coefficients = model.coef_[0]  # Use first class for multi-class
                
                fig_coef = self.create_feature_importance_plot(
                    coefficients,
                    feature_names,
                    f"Feature Coefficients - {selected_model}"
                )
                st.plotly_chart(fig_coef, use_container_width=True)
                
                # Coefficients table
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                st.subheader("📋 Feature Coefficients Table")
                st.dataframe(coef_df, use_container_width=True)
                
            else:
                st.info("This model doesn't provide built-in feature importance or coefficients.")
        
        # Explanation tips
        st.subheader("💡 Interpretation Tips")
        with st.expander("How to interpret explanations"):
            st.markdown("""
            **SHAP Values:**
            - Positive SHAP values increase the prediction
            - Negative SHAP values decrease the prediction
            - Larger absolute values indicate stronger influence
            
            **LIME Explanations:**
            - Shows local feature importance for individual predictions
            - Green bars typically indicate positive influence
            - Red bars typically indicate negative influence
            
            **Partial Dependence:**
            - Shows average effect of a feature across all instances
            - Helps understand global feature behavior
            - Steep slopes indicate strong feature influence
            
            **Feature Importance:**
            - Shows overall feature relevance for the model
            - Higher values indicate more important features
            - Based on model's internal importance calculations
            """)
