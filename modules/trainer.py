"""
Model Training Module for MatrixLab AI Studio
Enhanced version with comprehensive training capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Advanced model training with comprehensive evaluation"""

    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

        self.classification_algorithms = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Support Vector Machine": SVC(random_state=42, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Naive Bayes": GaussianNB()
        }

        self.regression_algorithms = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
            "Support Vector Regression": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Ridge Regression": Ridge(random_state=42),
            "Lasso Regression": Lasso(random_state=42)
        }

        self.param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            "Support Vector Machine": {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }

    def detect_problem_type(self, y: pd.Series) -> str:
        """Detect if problem is classification or regression"""
        if pd.api.types.is_numeric_dtype(y):
            unique_values = y.nunique()
            if unique_values <= 10 and unique_values > 1:
                return "Classification"
            else:
                return "Regression"
        else:
            return "Classification"

    def prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, LabelEncoder], Optional[LabelEncoder]]:
        """Prepare data for training"""
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.columns.difference(numeric_cols)

        # Fill missing values
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        if len(categorical_cols) > 0:
            X[categorical_cols] = X[categorical_cols].astype(str)
            for col in categorical_cols:
                mode_val = X[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
                X[col].fillna(fill_val, inplace=True)

        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # Encode target if categorical
        target_encoder = None
        if not pd.api.types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))

        return X.values, y, feature_cols, label_encoders, target_encoder

    def train_single_model(self, X: np.ndarray, y: np.ndarray, algorithm: str, problem_type: str, 
                          test_size: float = 0.2, use_cv: bool = True, hyperparameter_tuning: bool = False) -> Dict:
        """Train a single model with comprehensive evaluation"""
        
        # Select algorithm
        if problem_type == "Classification":
            model = self.classification_algorithms[algorithm]
        else:
            model = self.regression_algorithms[algorithm]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if problem_type == "Classification" else None
        )
        
        # Feature scaling for certain algorithms
        scaler = None
        if algorithm in ["Support Vector Machine", "Support Vector Regression", "K-Nearest Neighbors", "Logistic Regression"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Hyperparameter tuning
        if hyperparameter_tuning and algorithm in self.param_grids:
            param_grid = self.param_grids[algorithm]
            scoring = 'accuracy' if problem_type == "Classification" else 'r2'
            
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring=scoring, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model.fit(X_train, y_train)
            best_params = None
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if problem_type == "Classification":
            metrics = self._calculate_classification_metrics(y_test, y_pred, model, X_test)
        else:
            metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Cross-validation
        cv_scores = None
        if use_cv:
            scoring = 'accuracy' if problem_type == "Classification" else 'r2'
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
        
        # Training time
        training_time = datetime.datetime.now()
        
        result = {
            'model': model,
            'scaler': scaler,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'best_params': best_params,
            'training_time': training_time,
            'problem_type': problem_type,
            'algorithm': algorithm
        }
        
        return result

    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model: Any, X_test: np.ndarray) -> Dict:
        """Calculate comprehensive classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                    metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except:
                metrics['roc_auc'] = None
        
        return metrics

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive regression metrics"""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else None
        }

    def train_multiple_models(self, X: np.ndarray, y: np.ndarray, problem_type: str, 
                             selected_algorithms: List[str], **kwargs) -> Dict[str, Dict]:
        """Train multiple models and compare performance"""
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, algorithm in enumerate(selected_algorithms):
            status_text.text(f"Training {algorithm}...")
            
            try:
                result = self.train_single_model(X, y, algorithm, problem_type, **kwargs)
                results[algorithm] = result
                
                # Display quick metrics
                if problem_type == "Classification":
                    accuracy = result['metrics']['accuracy']
                    st.success(f"✅ {algorithm}: Accuracy = {accuracy:.4f}")
                else:
                    r2 = result['metrics']['r2_score']
                    st.success(f"✅ {algorithm}: R² = {r2:.4f}")
                    
            except Exception as e:
                st.error(f"❌ Error training {algorithm}: {str(e)}")
                
            progress_bar.progress((i + 1) / len(selected_algorithms))
        
        status_text.text("Training completed!")
        return results

    def create_model_comparison_chart(self, results: Dict[str, Dict], problem_type: str) -> go.Figure:
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
                marker_color=px.colors.qualitative.Set3[:len(model_names)],
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

    def create_metrics_comparison_table(self, results: Dict[str, Dict], problem_type: str) -> pd.DataFrame:
        """Create comprehensive metrics comparison table"""
        if problem_type == "Classification":
            metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean', 'cv_std']
        else:
            metrics_to_show = ['r2_score', 'mse', 'rmse', 'mae', 'cv_mean', 'cv_std']
        
        comparison_data = []
        for model_name, result in results.items():
            row = {'Model': model_name}
            for metric in metrics_to_show:
                if metric in result['metrics']:
                    row[metric.replace('_', ' ').title()] = result['metrics'][metric]
                else:
                    row[metric.replace('_', ' ').title()] = None
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)

    def render_training_tab(self, df: Optional[pd.DataFrame] = None):
        """Render the complete training tab"""
        st.header("🎯 Model Training")
        
        if df is None:
            st.warning("Please upload and preprocess data first.")
            return
        
        # Data preparation section
        st.subheader("📊 Data Preparation")
        
        # Feature and target selection
        all_columns = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Select target column:",
                all_columns,
                index=len(all_columns)-1 if all_columns else 0
            )
        
        with col2:
            feature_columns = st.multiselect(
                "Select feature columns:",
                [col for col in all_columns if col != target_column],
                default=[col for col in all_columns if col != target_column]
            )
        
        if not feature_columns:
            st.warning("Please select at least one feature column.")
            return
        
        # Detect problem type
        problem_type = self.detect_problem_type(df[target_column])
        st.info(f"Detected problem type: **{problem_type}**")
        
        # Training options
        st.subheader("🎯 Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if problem_type == "Classification":
                available_algorithms = list(self.classification_algorithms.keys())
            else:
                available_algorithms = list(self.regression_algorithms.keys())
            
            selected_algorithms = st.multiselect(
                "Select algorithms to train:",
                available_algorithms,
                default=available_algorithms[:3]  # Default to first 3
            )
        
        with col2:
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
            use_cv = st.checkbox("Use cross-validation", value=True)
            hyperparameter_tuning = st.checkbox("Enable hyperparameter tuning", value=False)
        
        # Training execution
        if st.button("🚀 Start Training", type="primary"):
            if not selected_algorithms:
                st.warning("Please select at least one algorithm.")
                return
            
            try:
                # Prepare data
                with st.spinner("Preparing data..."):
                    X, y, feature_names, label_encoders, target_encoder = self.prepare_data(
                        df, target_column, feature_columns
                    )
                
                # Store preprocessing info in session state
                st.session_state.preprocessing_info = {
                    'feature_names': feature_names,
                    'label_encoders': label_encoders,
                    'target_encoder': target_encoder,
                    'target_column': target_column
                }
                
                # Train models
                st.subheader("🔄 Training Progress")
                results = self.train_multiple_models(
                    X, y, problem_type, selected_algorithms,
                    test_size=test_size, use_cv=use_cv, 
                    hyperparameter_tuning=hyperparameter_tuning
                )
                
                # Store results in session state
                st.session_state.training_results = results
                
                # Display results
                st.subheader("📊 Training Results")
                
                # Model comparison chart
                if len(results) > 1:
                    fig = self.create_model_comparison_chart(results, problem_type)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Metrics comparison table
                comparison_df = self.create_metrics_comparison_table(results, problem_type)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Best model highlight
                if problem_type == "Classification":
                    best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
                    st.success(f"🏆 Best Model: **{best_model[0]}** with accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
                else:
                    best_model = max(results.items(), key=lambda x: x[1]['metrics']['r2_score'])
                    st.success(f"🏆 Best Model: **{best_model[0]}** with R²: {best_model[1]['metrics']['r2_score']:.4f}")
                
                # Individual model details
                st.subheader("🔍 Model Details")
                
                for model_name, result in results.items():
                    with st.expander(f"📋 {model_name} Details"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Metrics:**")
                            for metric, value in result['metrics'].items():
                                if value is not None:
                                    st.write(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
                        
                        with col2:
                            if result['best_params']:
                                st.write("**Best Parameters:**")
                                for param, value in result['best_params'].items():
                                    st.write(f"- {param}: {value}")
                            
                            if result['cv_scores'] is not None:
                                st.write("**Cross-Validation Scores:**")
                                cv_scores = result['cv_scores']
                                st.write(f"- Mean: {cv_scores.mean():.4f}")
                                st.write(f"- Std: {cv_scores.std():.4f}")
                
                st.success("🎉 Training completed successfully!")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.exception(e)
