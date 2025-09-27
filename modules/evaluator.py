"""
Model Evaluation Module for MatrixLab AI Studio
Comprehensive model evaluation with visualizations and metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             labels: Optional[List[str]] = None) -> go.Figure:
        """Create interactive confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        
        # Calculate percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{cm[i][j]}<br>({cm_normalized[i][j]:.2%})",
                        showarrow=False,
                        font=dict(color="white" if cm_normalized[i][j] > 0.5 else "black")
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            annotations=annotations,
            width=500,
            height=500
        )
        
        return fig
    
    def get_classification_report_df(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Get detailed classification report as DataFrame"""
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Convert to DataFrame
            df = pd.DataFrame(report).transpose()
            
            # Round numerical values
            numeric_columns = ['precision', 'recall', 'f1-score']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].round(4)
            
            return df
            
        except Exception as e:
            st.error(f"Error generating classification report: {str(e)}")
            return pd.DataFrame()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      pos_label: int = 1) -> go.Figure:
        """Plot ROC curve for binary classification"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.4f})',
                line=dict(color='blue', width=2)
            ))
            
            # Diagonal line (random classifier)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                width=500,
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error plotting ROC curve: {str(e)}")
            return None
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                   pos_label: int = 1) -> go.Figure:
        """Plot Precision-Recall curve"""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba, pos_label=pos_label)
            avg_precision = average_precision_score(y_true, y_proba, pos_label=pos_label)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'PR Curve (AP = {avg_precision:.4f})',
                line=dict(color='green', width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                width=500,
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error plotting PR curve: {str(e)}")
            return None
    
    def plot_feature_importance(self, importance_values: np.ndarray, 
                               feature_names: List[str], 
                               title: str = "Feature Importance") -> go.Figure:
        """Plot feature importance"""
        if len(importance_values) != len(feature_names):
            st.error("Length mismatch between importance values and feature names")
            return None
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color=self.color_palette[0]
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Features',
            height=max(400, len(feature_names) * 25),
            margin=dict(l=150)
        )
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      title: str = "Residual Analysis") -> go.Figure:
        """Plot residuals for regression models"""
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Residuals vs Predicted",
                "Residuals Distribution",
                "Q-Q Plot",
                "Actual vs Predicted"
            ]
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color=self.color_palette[0])
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residuals histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residuals',
                marker_color=self.color_palette[1]
            ),
            row=1, col=2
        )
        
        # Q-Q plot (simplified)
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color=self.color_palette[2])
            ),
            row=2, col=1
        )
        
        # Add diagonal line for Q-Q plot
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=theoretical_quantiles * np.std(residuals) + np.mean(residuals),
                mode='lines',
                name='Perfect Fit',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color=self.color_palette[3])
            ),
            row=2, col=2
        )
        
        # Add perfect prediction line
        min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_model_comparison_chart(self, results: Dict[str, Dict], 
                                    problem_type: str) -> go.Figure:
        """Create comparison chart for multiple models"""
        model_names = list(results.keys())
        
        if problem_type == "Classification":
            metric_name = 'accuracy'
            title = "Model Comparison - Accuracy"
        else:
            metric_name = 'r2_score'
            title = "Model Comparison - R² Score"
        
        values = [results[model]['metrics'][metric_name] for model in model_names]
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=values,
                marker_color=self.color_palette[:len(model_names)],
                text=[f"{v:.4f}" for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Models",
            yaxis_title=metric_name.replace('_', ' ').title(),
            xaxis_tickangle=-45
        )
        
        return fig
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else None
        }
    
    def render_evaluation_tab(self, training_results: Optional[Dict] = None) -> None:
        """Render comprehensive model evaluation interface"""
        st.header("📈 Model Evaluation")
        
        if not training_results:
            st.warning("⚠️ No training results found. Please train models first.")
            return
        
        # Model selection for detailed evaluation
        st.subheader("🔍 Select Model for Detailed Evaluation")
        model_names = list(training_results.keys())
        selected_model = st.selectbox("Choose model:", model_names)
        
        if not selected_model:
            return
        
        result = training_results[selected_model]
        problem_type = result['problem_type']
        
        # Model overview
        st.subheader(f"📊 {selected_model} - {problem_type}")
        
        # Basic metrics display
        metrics = result['metrics']
        if problem_type == "Classification":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
            with col2:
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            with col3:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            with col4:
                if metrics.get('mape') is not None:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
        
        # Detailed evaluation based on problem type
        if problem_type == "Classification":
            self._render_classification_evaluation(result)
        else:
            self._render_regression_evaluation(result)
        
        # Cross-validation results
        if result.get('cv_scores') is not None:
            st.subheader("🎯 Cross-Validation Results")
            cv_scores = result['cv_scores']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CV Mean", f"{cv_scores.mean():.4f}")
                st.metric("CV Std", f"{cv_scores.std():.4f}")
            
            with col2:
                # CV scores plot
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                        y=cv_scores,
                        marker_color=self.color_palette[0]
                    )
                ])
                fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red")
                fig.update_layout(title="Cross-Validation Scores", height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        model = result['model']
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            feature_names = st.session_state.get('preprocessing_info', {}).get('feature_names', [])
            if feature_names:
                fig = self.plot_feature_importance(
                    model.feature_importances_, 
                    feature_names,
                    f"Feature Importance - {selected_model}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        if len(training_results) > 1:
            st.subheader("⚖️ Model Comparison")
            fig = self.create_model_comparison_chart(training_results, problem_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            comparison_df = self._create_comparison_table(training_results, problem_type)
            st.dataframe(comparison_df, use_container_width=True)
    
    def _render_classification_evaluation(self, result: Dict) -> None:
        """Render classification-specific evaluation"""
        y_true = result['y_test']
        y_pred = result['y_pred']
        model = result['model']
        X_test = result['X_test']
        
        # Confusion Matrix
        st.subheader("🎯 Confusion Matrix")
        fig_cm = self.plot_confusion_matrix(y_true, y_pred)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification Report
        st.subheader("📋 Classification Report")
        report_df = self.get_classification_report_df(y_true, y_pred)
        if not report_df.empty:
            st.dataframe(report_df, use_container_width=True)
        
        # ROC and PR curves for binary classification
        if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 ROC Curve")
                    fig_roc = self.plot_roc_curve(y_true, y_proba)
                    if fig_roc:
                        st.plotly_chart(fig_roc, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Precision-Recall Curve")
                    fig_pr = self.plot_precision_recall_curve(y_true, y_proba)
                    if fig_pr:
                        st.plotly_chart(fig_pr, use_container_width=True)
                        
            except Exception as e:
                st.warning(f"Could not generate probability curves: {str(e)}")
    
    def _render_regression_evaluation(self, result: Dict) -> None:
        """Render regression-specific evaluation"""
        y_true = result['y_test']
        y_pred = result['y_pred']
        
        # Residual Analysis
        st.subheader("📊 Residual Analysis")
        fig_residuals = self.plot_residuals(y_true, y_pred)
        st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Additional regression metrics
        additional_metrics = self.calculate_regression_metrics(y_true, y_pred)
        
        st.subheader("📈 Additional Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            for i, (metric, value) in enumerate(list(additional_metrics.items())[:3]):
                if value is not None:
                    st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
        
        with col2:
            for i, (metric, value) in enumerate(list(additional_metrics.items())[3:]):
                if value is not None:
                    if metric == 'mape':
                        st.metric(metric.upper(), f"{value:.2f}%")
                    else:
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
    
    def _create_comparison_table(self, results: Dict[str, Dict], problem_type: str) -> pd.DataFrame:
        """Create comprehensive comparison table"""
        comparison_data = []
        
        for model_name, result in results.items():
            row = {'Model': model_name}
            metrics = result['metrics']
            
            if problem_type == "Classification":
                row.update({
                    'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                    'Precision': f"{metrics.get('precision', 0):.4f}",
                    'Recall': f"{metrics.get('recall', 0):.4f}",
                    'F1-Score': f"{metrics.get('f1_score', 0):.4f}"
                })
            else:
                row.update({
                    'R² Score': f"{metrics.get('r2_score', 0):.4f}",
                    'MAE': f"{metrics.get('mae', 0):.4f}",
                    'RMSE': f"{metrics.get('rmse', 0):.4f}"
                })
            
            if result.get('cv_scores') is not None:
                cv_scores = result['cv_scores']
                row.update({
                    'CV Mean': f"{cv_scores.mean():.4f}",
                    'CV Std': f"{cv_scores.std():.4f}"
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
