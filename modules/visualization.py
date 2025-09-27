"""
Visualization Module for MatrixLab AI Studio
Advanced plotting and chart creation utilities
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """Advanced visualization utilities for ML workflows"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.sequential_colors = px.colors.sequential.Viridis
        self.diverging_colors = px.colors.diverging.RdBu
    
    def create_correlation_heatmap(self, data: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
        """Create interactive correlation heatmap"""
        if data.select_dtypes(include=[np.number]).shape[1] < 2:
            return None
        
        corr_matrix = data.select_dtypes(include=[np.number]).corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.where(~mask)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=600
        )
        
        return fig
    
    def create_distribution_plot(self, data: pd.DataFrame, column: str, 
                               plot_type: str = "histogram") -> go.Figure:
        """Create distribution plots (histogram, box, violin)"""
        if column not in data.columns:
            return None
        
        fig = go.Figure()
        
        if plot_type == "histogram":
            fig.add_trace(go.Histogram(
                x=data[column],
                nbinsx=30,
                name=column,
                marker_color=self.color_palette[0],
                opacity=0.7
            ))
            
            # Add mean and median lines
            mean_val = data[column].mean()
            median_val = data[column].median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                         annotation_text=f"Median: {median_val:.2f}")
            
        elif plot_type == "box":
            fig.add_trace(go.Box(
                y=data[column],
                name=column,
                marker_color=self.color_palette[0]
            ))
            
        elif plot_type == "violin":
            fig.add_trace(go.Violin(
                y=data[column],
                name=column,
                box_visible=True,
                line_color=self.color_palette[0]
            ))
        
        fig.update_layout(
            title=f"{plot_type.title()} Plot - {column}",
            xaxis_title=column if plot_type == "histogram" else "",
            yaxis_title="Frequency" if plot_type == "histogram" else column
        )
        
        return fig
    
    def create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str, 
                          color_col: Optional[str] = None, size_col: Optional[str] = None) -> go.Figure:
        """Create interactive scatter plot"""
        fig_data = dict(x=data[x_col], y=data[y_col], mode='markers')
        
        if color_col and color_col in data.columns:
            fig_data['marker'] = dict(
                color=data[color_col],
                colorscale=self.sequential_colors,
                colorbar=dict(title=color_col),
                showscale=True
            )
        else:
            fig_data['marker'] = dict(color=self.color_palette[0])
        
        if size_col and size_col in data.columns:
            if 'marker' not in fig_data:
                fig_data['marker'] = {}
            fig_data['marker']['size'] = data[size_col]
            fig_data['marker']['sizemode'] = 'diameter'
            fig_data['marker']['sizeref'] = 2. * max(data[size_col]) / (40.**2)
        
        fig = go.Figure(data=go.Scatter(**fig_data))
        
        # Add trendline
        try:
            z = np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=p(data[x_col]),
                mode='lines',
                name='Trendline',
                line=dict(dash='dash', color='red')
            ))
        except:
            pass
        
        fig.update_layout(
            title=f"Scatter Plot: {x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def create_categorical_plot(self, data: pd.DataFrame, column: str, 
                              plot_type: str = "bar") -> go.Figure:
        """Create categorical data visualizations"""
        if column not in data.columns:
            return None
        
        value_counts = data[column].value_counts()
        
        if plot_type == "bar":
            fig = go.Figure(data=[
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker_color=self.color_palette[:len(value_counts)],
                    text=value_counts.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Value Counts - {column}",
                xaxis_title=column,
                yaxis_title="Count",
                xaxis_tickangle=-45
            )
            
        elif plot_type == "pie":
            fig = go.Figure(data=[
                go.Pie(
                    labels=value_counts.index,
                    values=value_counts.values,
                    hole=0.3,
                    marker_colors=self.color_palette[:len(value_counts)]
                )
            ])
            
            fig.update_layout(
                title=f"Distribution - {column}",
                annotations=[dict(text=column, x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
        
        return fig
    
    def create_pairplot_matrix(self, data: pd.DataFrame, columns: List[str], 
                             max_cols: int = 4) -> go.Figure:
        """Create pairplot matrix for numerical columns"""
        # Limit columns for performance
        columns = columns[:max_cols]
        n_cols = len(columns)
        
        if n_cols < 2:
            return None
        
        fig = make_subplots(
            rows=n_cols, 
            cols=n_cols,
            subplot_titles=[f"{col1} vs {col2}" if i != j else f"{col1} distribution" 
                           for i, col1 in enumerate(columns) for j, col2 in enumerate(columns)]
        )
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    # Diagonal: histogram
                    fig.add_trace(
                        go.Histogram(x=data[col1], name=f"{col1}", showlegend=False,
                                   marker_color=self.color_palette[i % len(self.color_palette)]),
                        row=i+1, col=j+1
                    )
                else:
                    # Off-diagonal: scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=data[col2], 
                            y=data[col1], 
                            mode='markers',
                            name=f"{col1} vs {col2}",
                            showlegend=False,
                            marker=dict(size=4, color=self.color_palette[(i+j) % len(self.color_palette)])
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            title="Pairplot Matrix",
            height=200 * n_cols,
            showlegend=False
        )
        
        return fig
    
    def create_missing_values_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create missing values visualization"""
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if missing_data.empty:
            # No missing values
            fig = go.Figure()
            fig.add_annotation(
                text="No Missing Values Found! 🎉",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=24, color="green")
            )
            fig.update_layout(
                title="Missing Values Analysis",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Calculate percentages
        missing_pct = (missing_data / len(data)) * 100
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Missing Count", "Missing Percentage"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Missing count bar chart
        fig.add_trace(
            go.Bar(
                x=missing_data.index,
                y=missing_data.values,
                name="Count",
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Missing percentage bar chart
        fig.add_trace(
            go.Bar(
                x=missing_pct.index,
                y=missing_pct.values,
                name="Percentage",
                marker_color='orange',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Missing Values Analysis",
            height=400,
            showlegend=False
        )
        
        # Update x-axis labels
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        
        return fig
    
    def create_outlier_detection_plot(self, data: pd.DataFrame, columns: List[str]) -> go.Figure:
        """Create outlier detection visualization using box plots"""
        if not columns:
            return None
        
        # Select only numeric columns
        numeric_columns = [col for col in columns if col in data.select_dtypes(include=[np.number]).columns]
        
        if not numeric_columns:
            return None
        
        fig = go.Figure()
        
        for i, col in enumerate(numeric_columns):
            fig.add_trace(go.Box(
                y=data[col],
                name=col,
                marker_color=self.color_palette[i % len(self.color_palette)],
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Outlier Detection - Box Plots",
            yaxis_title="Values",
            xaxis_title="Features"
        )
        
        return fig
    
    def create_feature_distribution_grid(self, data: pd.DataFrame, columns: List[str], 
                                       max_cols: int = 3) -> go.Figure:
        """Create grid of distribution plots"""
        if not columns:
            return None
        
        # Limit columns for performance
        columns = columns[:min(9, len(columns))]  # Max 9 subplots
        
        n_cols = min(max_cols, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=columns,
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(columns):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            if data[col].dtype in ['int64', 'float64']:
                # Histogram for numerical columns
                fig.add_trace(
                    go.Histogram(
                        x=data[col], 
                        name=col, 
                        showlegend=False,
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ),
                    row=row, col=col_pos
                )
            else:
                # Bar chart for categorical columns (top 10 values)
                value_counts = data[col].value_counts().head(10)
                fig.add_trace(
                    go.Bar(
                        x=value_counts.index, 
                        y=value_counts.values, 
                        name=col, 
                        showlegend=False,
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ),
                    row=row, col=col_pos
                )
        
        fig.update_layout(
            title="Feature Distribution Grid",
            height=300 * n_rows,
            showlegend=False
        )
        
        return fig
    
    def create_target_analysis_plot(self, data: pd.DataFrame, target_col: str, 
                                  feature_cols: List[str]) -> go.Figure:
        """Create target variable analysis plots"""
        if target_col not in data.columns:
            return None
        
        # Determine if target is categorical or numerical
        is_categorical = data[target_col].dtype == 'object' or data[target_col].nunique() <= 10
        
        if is_categorical:
            # Categorical target - create stacked bar charts for features
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Target Distribution",
                    f"{feature_cols[0] if feature_cols else 'Feature'} vs Target",
                    f"{feature_cols[1] if len(feature_cols) > 1 else 'Feature'} vs Target",
                    "Target Balance"
                ]
            )
            
            # Target distribution
            target_counts = data[target_col].value_counts()
            fig.add_trace(
                go.Bar(x=target_counts.index, y=target_counts.values, 
                      marker_color=self.color_palette[0]),
                row=1, col=1
            )
            
            # Feature vs target analysis
            if feature_cols:
                for i, feature in enumerate(feature_cols[:2]):
                    row = 1 if i == 1 else 2
                    col = 2 if i == 1 else 1
                    
                    if data[feature].dtype in ['int64', 'float64']:
                        # Box plot for numerical feature
                        for j, target_val in enumerate(data[target_col].unique()):
                            fig.add_trace(
                                go.Box(
                                    y=data[data[target_col] == target_val][feature],
                                    name=f"{target_val}",
                                    marker_color=self.color_palette[j % len(self.color_palette)]
                                ),
                                row=row, col=col
                            )
                    else:
                        # Stacked bar for categorical feature
                        crosstab = pd.crosstab(data[feature], data[target_col])
                        for j, target_val in enumerate(crosstab.columns):
                            fig.add_trace(
                                go.Bar(
                                    x=crosstab.index,
                                    y=crosstab[target_val],
                                    name=f"{target_val}",
                                    marker_color=self.color_palette[j % len(self.color_palette)]
                                ),
                                row=row, col=col
                            )
            
            # Target balance pie chart
            fig.add_trace(
                go.Pie(
                    labels=target_counts.index,
                    values=target_counts.values,
                    hole=0.3,
                    marker_colors=self.color_palette[:len(target_counts)]
                ),
                row=2, col=2
            )
            
        else:
            # Numerical target - create regression-style plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Target Distribution",
                    f"{feature_cols[0] if feature_cols else 'Feature'} vs Target",
                    f"{feature_cols[1] if len(feature_cols) > 1 else 'Feature'} vs Target",
                    "Target Statistics"
                ]
            )
            
            # Target histogram
            fig.add_trace(
                go.Histogram(x=data[target_col], marker_color=self.color_palette[0]),
                row=1, col=1
            )
            
            # Feature scatter plots
            if feature_cols:
                for i, feature in enumerate(feature_cols[:2]):
                    row = 1 if i == 1 else 2
                    col = 2 if i == 1 else 1
                    
                    if data[feature].dtype in ['int64', 'float64']:
                        fig.add_trace(
                            go.Scatter(
                                x=data[feature],
                                y=data[target_col],
                                mode='markers',
                                marker=dict(color=self.color_palette[i], size=4)
                            ),
                            row=row, col=col
                        )
        
        fig.update_layout(
            title=f"Target Analysis - {target_col}",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_model_performance_plot(self, results: Dict[str, Dict], 
                                    problem_type: str) -> go.Figure:
        """Create model performance comparison plot"""
        model_names = list(results.keys())
        
        if problem_type == "Classification":
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            metric_values = {metric: [] for metric in metrics}
            
            for model_name in model_names:
                model_metrics = results[model_name]['metrics']
                for metric in metrics:
                    metric_values[metric].append(model_metrics.get(metric, 0))
            
            fig = go.Figure()
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=model_names,
                    y=metric_values[metric],
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ))
            
            fig.update_layout(
                title="Classification Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group'
            )
            
        else:
            # Regression metrics
            metrics = ['r2_score', 'mae', 'rmse']
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[metric.replace('_', ' ').title() for metric in metrics]
            )
            
            for i, metric in enumerate(metrics):
                values = [results[model]['metrics'].get(metric, 0) for model in model_names]
                
                fig.add_trace(
                    go.Bar(
                        x=model_names,
                        y=values,
                        name=metric.replace('_', ' ').title(),
                        marker_color=self.color_palette[i % len(self.color_palette)],
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title="Regression Model Performance Comparison",
                height=400
            )
        
        return fig
