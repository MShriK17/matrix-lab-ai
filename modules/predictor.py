"""
Enhanced Prediction Module for MatrixLab AI Studio
Handles batch CSV predictions and real-time manual input
"""

import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


class ModelPredictor:
    """Enhanced batch & real-time prediction helper"""

    def __init__(self) -> None:
        self.models_dir = "models"

    # ───────────────────────── helpers ───────────────────────── #
    @staticmethod
    def _warn(msg: str) -> None:
        st.warning(msg, icon="⚠️")

    @staticmethod
    def _success(msg: str) -> None:
        st.success(msg, icon="✅")

    @staticmethod
    def _error(msg: str) -> None:
        st.error(msg, icon="❌")

    # ─────────────────── load / preprocess ───────────────────── #
    def load_model_for_prediction(self, path: str) -> Tuple[Any, Dict, Optional[Any]]:
        """Load model with metadata and scaler"""
        try:
            data = joblib.load(path)
            if isinstance(data, dict):
                return data.get("model"), data.get("metadata", {}), data.get("scaler")
            # legacy format
            return data, {}, None
        except Exception as e:
            self._error(f"Failed to load model: {str(e)}")
            return None, {}, None

    def preprocess(self,
                   df: pd.DataFrame,
                   feats: List[str],
                   label_encoders: Dict[str, Any] = None,
                   scaler: Any = None) -> Optional[np.ndarray]:
        """Preprocess data for prediction"""
        try:
            # Select features
            if feats:
                missing = set(feats) - set(df.columns)
                if missing:
                    self._warn(f"Missing required features: {missing}")
                    return None
                X = df[feats].copy()
            else:
                X = df.copy()

            # Handle missing values
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            
            # Fill missing values
            if len(numeric_cols) > 0:
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
            
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_val = X[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
                    X[col].fillna(fill_val, inplace=True)

            # Label encode categorical variables
            if label_encoders:
                for col, enc in label_encoders.items():
                    if col in X.columns:
                        try:
                            X[col] = enc.transform(X[col].astype(str))
                        except ValueError as e:
                            # Handle unseen categories
                            st.warning(f"Unseen categories in {col}. Using most frequent class.")
                            X[col] = enc.transform([enc.classes_[0]] * len(X))

            # Scale features
            if scaler is not None:
                X = scaler.transform(X)

            return X if isinstance(X, np.ndarray) else X.values

        except Exception as exc:
            self._error(f"Pre-processing error: {exc}")
            return None

    # ─────────────────────── predictions ─────────────────────── #
    @staticmethod
    def _predict(model: Any,
                 X: np.ndarray,
                 target_enc: Any = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with probability estimates"""
        try:
            preds = model.predict(X)
            
            # Inverse transform target if encoded
            if target_enc is not None:
                preds = target_enc.inverse_transform(preds)

            # Get prediction probabilities
            probs = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X)
                except Exception:
                    pass
            
            return preds, probs
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

    # ──────────────────────── UI main ────────────────────────── #
    def render_prediction_tab(self, training_results: Dict = None) -> None:
        st.header("🔮 Model Predictions")
        st.markdown("Make predictions using trained models or saved models.")

        # ---- collect available models
        choices: Dict[str, Dict] = {}
        
        # Add session models
        if training_results:
            for nm, d in training_results.items():
                choices[f"Session ⟩ {nm}"] = {"type": "session", "data": d}

        # Add saved models
        if os.path.isdir(self.models_dir):
            for f in os.listdir(self.models_dir):
                if f.endswith(".joblib"):
                    pth = os.path.join(self.models_dir, f)
                    mdl, meta, scl = self.load_model_for_prediction(pth)
                    if mdl is not None:
                        disp = meta.get("model_name", f.replace(".joblib", ""))
                        choices[f"Saved ⟩ {disp}"] = {
                            "type": "saved", "model": mdl, "meta": meta, "scaler": scl
                        }

        if not choices:
            self._warn("No models available. Please train a model first.")
            return

        # Model selection
        st.subheader("🎯 Model Selection")
        sel_key = st.selectbox("Choose a model for predictions:", list(choices.keys()))
        info = choices[sel_key]

        # ---- Model information display
        if info["type"] == "session":
            d = info["data"]
            st.info(f"**{d['algorithm']}** • {d['problem_type']} • Session Model")
            
            # Display model metrics
            metrics = d.get('metrics', {})
            if metrics:
                st.subheader("📊 Model Performance")
                cols = st.columns(len(metrics))
                for i, (metric, value) in enumerate(metrics.items()):
                    if value is not None:
                        cols[i].metric(metric.replace('_', ' ').title(), f"{value:.4f}")
        else:
            meta = info["meta"]
            st.info(f"**{meta.get('algorithm','Unknown')}** • {meta.get('problem_type','Unknown')} • v{meta.get('version','1.0')}")
            
            # Display model information
            if meta.get('training_time'):
                st.write(f"**Trained on:** {meta.get('training_time')}")
            if meta.get('description'):
                st.write(f"**Description:** {meta.get('description')}")

        # Prediction mode selection
        st.subheader("📋 Prediction Mode")
        mode = st.radio(
            "Select prediction mode:",
            ["Batch CSV Upload", "Real-time Manual Input"],
            horizontal=True
        )

        # Render appropriate interface
        if mode == "Batch CSV Upload":
            self._ui_batch(info)
        else:
            self._ui_realtime(info)

    # ─────────────────── batch CSV UI ────────────────────────── #
    def _ui_batch(self, info: Dict) -> None:
        st.subheader("📁 Batch CSV Predictions")
        st.markdown("Upload a CSV file with the same feature columns as your training data.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch predictions",
            type="csv",
            help="Make sure the CSV contains the same feature columns used during training"
        )
        
        if not uploaded_file:
            st.info("👆 Please upload a CSV file to make batch predictions")
            return

        try:
            # Load and preview data
            df = pd.read_csv(uploaded_file)
            st.write("📊 **Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", f"{df.shape[1]:,}")
            with col3:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

            # Extract model components
            model, feats, encs, scaler, tgt_enc, problem, name = self._extract_model_parts(info)

            # Prediction button
            if st.button("🚀 Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    # Preprocess data
                    X = self.preprocess(df, feats, encs, scaler)
                    if X is None:
                        return
                    
                    # Make predictions
                    preds, probs = self._predict(model, X, tgt_enc)
                    if preds is None:
                        return

                    # Prepare output
                    output_df = df.copy()
                    output_df["Prediction"] = preds
                    
                    # Add confidence scores
                    if probs is not None:
                        if problem == "Classification":
                            output_df["Confidence"] = probs.max(axis=1)
                            # Add class probabilities for binary classification
                            if probs.shape[1] == 2:
                                output_df["Prob_Class_0"] = probs[:, 0]
                                output_df["Prob_Class_1"] = probs[:, 1]
                        else:
                            # For regression, we might not have probabilities
                            pass
                    
                    self._success(f"Successfully generated {len(preds)} predictions!")
                    
                    # Display results
                    st.subheader("📈 Prediction Results")
                    st.dataframe(output_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📊 Prediction Summary")
                    if problem == "Classification":
                        pred_counts = pd.Series(preds).value_counts()
                        st.write("**Prediction Distribution:**")
                        st.bar_chart(pred_counts)
                        
                        if probs is not None:
                            avg_confidence = probs.max(axis=1).mean()
                            st.metric("Average Confidence", f"{avg_confidence:.3f}")
                    else:
                        st.metric("Mean Prediction", f"{np.mean(preds):.3f}")
                        st.metric("Prediction Std", f"{np.std(preds):.3f}")
                        st.metric("Min Prediction", f"{np.min(preds):.3f}")
                        st.metric("Max Prediction", f"{np.max(preds):.3f}")
                    
                    # Download button
                    csv_data = output_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Predictions CSV",
                        data=csv_data,
                        file_name=f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            self._error(f"Error processing file: {str(e)}")

    # ────────────────── real-time form UI ────────────────────── #
    def _ui_realtime(self, info: Dict) -> None:
        st.subheader("⚡ Real-time Single Prediction")
        st.markdown("Enter values manually to get instant predictions.")

        # Extract model components
        model, feats, encs, scaler, tgt_enc, problem, name = self._extract_model_parts(info)
        
        if not feats:
            self._warn("Feature information missing. Cannot build manual input form.")
            return

        # Create input form
        st.write("🔧 **Input Features:**")
        
        with st.form("realtime_prediction_form"):
            # Organize inputs in columns
            n_cols = min(3, len(feats))
            cols = st.columns(n_cols)
            vals: Dict[str, Any] = {}

            for i, feature in enumerate(feats):
                col = cols[i % n_cols]
                with col:
                    if feature in encs:  # Categorical feature
                        vals[feature] = st.selectbox(
                            f"🔤 {feature}",
                            list(encs[feature].classes_),
                            help=f"Select a value for {feature}"
                        )
                    else:  # Numerical feature
                        vals[feature] = st.number_input(
                            f"🔢 {feature}",
                            value=0.0,
                            help=f"Enter a numerical value for {feature}"
                        )

            # Prediction button
            predict_button = st.form_submit_button("🎯 Make Prediction", type="primary")

        # Process prediction
        if predict_button:
            try:
                # Create DataFrame from inputs
                input_df = pd.DataFrame([vals])
                
                # Preprocess
                X = self.preprocess(input_df, feats, encs, scaler)
                if X is None:
                    return
                
                # Make prediction
                preds, probs = self._predict(model, X, tgt_enc)
                if preds is None:
                    return

                # Display results
                st.subheader("🎯 Prediction Result")
                
                # Main prediction
                prediction_value = preds[0]
                st.success(f"**Predicted Value:** {prediction_value}")
                
                # Additional information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction", str(prediction_value))
                
                with col2:
                    if probs is not None:
                        confidence = probs.max()
                        st.metric("Confidence", f"{confidence:.3f}")
                
                # Probability details for classification
                if problem == "Classification" and probs is not None:
                    st.subheader("📊 Class Probabilities")
                    
                    # Create probability chart
                    if tgt_enc is not None:
                        class_names = tgt_enc.classes_
                    else:
                        class_names = [f"Class {i}" for i in range(len(probs[0]))]
                    
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probs[0]
                    })
                    
                    st.bar_chart(prob_df.set_index('Class'))
                    
                    # Show probability table
                    st.dataframe(prob_df, use_container_width=True)

            except Exception as e:
                self._error(f"Prediction failed: {str(e)}")

    # ─────────────────── utils / extract ─────────────────────── #
    @staticmethod
    def _extract_model_parts(info: Dict):
        """Extract model components from info dict"""
        if info["type"] == "session":
            # Session model
            d = info["data"]
            pre = st.session_state.get("preprocessing_info", {})
            return (
                d["model"],
                pre.get("feature_names", []),
                pre.get("label_encoders", {}),
                d.get("scaler"),
                pre.get("target_encoder"),
                d["problem_type"],
                d["algorithm"]
            )
        else:
            # Saved model
            meta = info["meta"]
            return (
                info["model"],
                meta.get("feature_names", []),
                meta.get("label_encoders", {}),
                info.get("scaler"),
                meta.get("target_encoder"),
                meta.get("problem_type", "Unknown"),
                meta.get("algorithm", "Unknown")
            )
