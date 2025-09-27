"""
Model Management Module for MatrixLab AI Studio
Handle saving, loading, and managing trained models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go

class ModelManager:
    """Comprehensive model management system"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / "models_metadata.json"
        
    def save_model_with_metadata(self, model_data: Dict, model_name: str, 
                                description: str = "", tags: List[str] = None) -> str:
        """Save model with comprehensive metadata"""
        try:
            # Generate version number
            existing_versions = self._get_existing_versions(model_name)
            new_version = max(existing_versions) + 0.1 if existing_versions else 1.0
            
            # Create filename
            filename = f"{model_name}_v{new_version:.1f}.joblib"
            filepath = self.models_dir / filename
            
            # Prepare comprehensive metadata
            preprocessing_info = st.session_state.get('preprocessing_info', {})
            
            metadata = {
                'model_name': model_name,
                'version': new_version,
                'algorithm': model_data['algorithm'],
                'problem_type': model_data['problem_type'],
                'metrics': model_data['metrics'],
                'training_time': model_data['training_time'].isoformat(),
                'description': description,
                'tags': tags or [],
                'best_params': model_data.get('best_params'),
                'cv_scores': model_data.get('cv_scores').tolist() if model_data.get('cv_scores') is not None else None,
                'feature_names': preprocessing_info.get('feature_names', []),
                'target_column': preprocessing_info.get('target_column'),
                'label_encoders': {},  # Note: actual encoders stored separately
                'file_size': 0,  # Will be updated after saving
                'created_by': "MatrixLab AI Studio",
                'data_shape': {
                    'n_samples': len(model_data.get('X_train', [])) + len(model_data.get('X_test', [])),
                    'n_features': len(preprocessing_info.get('feature_names', []))
                }
            }
            
            # Save model and metadata together
            save_data = {
                'model': model_data['model'],
                'scaler': model_data.get('scaler'),
                'metadata': metadata,
                'preprocessing_info': preprocessing_info
            }
            
            joblib.dump(save_data, filepath)
            
            # Update file size in metadata
            file_size = filepath.stat().st_size
            metadata['file_size'] = file_size
            save_data['metadata']['file_size'] = file_size
            joblib.dump(save_data, filepath)  # Re-save with updated file size
            
            # Update global metadata index
            self._update_metadata_index(metadata, filename)
            
            return filename
            
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, filename: str) -> Optional[Dict]:
        """Load model with all associated data"""
        try:
            filepath = self.models_dir / filename
            if not filepath.exists():
                st.error(f"Model file not found: {filename}")
                return None
            
            data = joblib.load(filepath)
            return data
            
        except Exception as e:
            st.error(f"Error loading model {filename}: {str(e)}")
            return None
    
    def delete_model(self, filename: str) -> bool:
        """Delete model file and update metadata"""
        try:
            filepath = self.models_dir / filename
            if filepath.exists():
                filepath.unlink()
                self._remove_from_metadata_index(filename)
                return True
            return False
            
        except Exception as e:
            st.error(f"Error deleting model: {str(e)}")
            return False
    
    def list_saved_models(self) -> List[Dict]:
        """List all saved models with metadata"""
        models = []
        
        if not self.models_dir.exists():
            return models
        
        for file in self.models_dir.glob("*.joblib"):
            try:
                data = joblib.load(file)
                
                if isinstance(data, dict) and 'metadata' in data:
                    metadata = data['metadata'].copy()
                    metadata['filename'] = file.name
                    metadata['filepath'] = str(file)
                    metadata['file_size_mb'] = metadata.get('file_size', 0) / (1024 * 1024)
                    models.append(metadata)
                    
            except Exception as e:
                st.warning(f"Could not load metadata for {file.name}: {str(e)}")
        
        return sorted(models, key=lambda x: x.get('training_time', ''), reverse=True)
    
    def export_model(self, filename: str, export_format: str = 'joblib') -> Optional[bytes]:
        """Export model in specified format"""
        try:
            data = self.load_model(filename)
            if not data:
                return None
            
            if export_format == 'joblib':
                # Return the joblib file content
                filepath = self.models_dir / filename
                return filepath.read_bytes()
            
            elif export_format == 'json_metadata':
                # Export only metadata as JSON
                metadata = data.get('metadata', {})
                return json.dumps(metadata, indent=2, default=str).encode()
            
            else:
                st.error(f"Unsupported export format: {export_format}")
                return None
                
        except Exception as e:
            st.error(f"Error exporting model: {str(e)}")
            return None
    
    def get_model_analytics(self) -> Dict[str, Any]:
        """Get analytics about saved models"""
        models = self.list_saved_models()
        
        if not models:
            return {}
        
        analytics = {
            'total_models': len(models),
            'algorithms': {},
            'problem_types': {},
            'total_size_mb': 0,
            'avg_accuracy': 0,
            'best_model': None,
            'recent_models': []
        }
        
        accuracies = []
        
        for model in models:
            # Algorithm distribution
            algorithm = model.get('algorithm', 'Unknown')
            analytics['algorithms'][algorithm] = analytics['algorithms'].get(algorithm, 0) + 1
            
            # Problem type distribution
            problem_type = model.get('problem_type', 'Unknown')
            analytics['problem_types'][problem_type] = analytics['problem_types'].get(problem_type, 0) + 1
            
            # Total size
            analytics['total_size_mb'] += model.get('file_size_mb', 0)
            
            # Accuracy tracking
            metrics = model.get('metrics', {})
            if 'accuracy' in metrics:
                accuracies.append(metrics['accuracy'])
            elif 'r2_score' in metrics:
                accuracies.append(metrics['r2_score'])
        
        # Calculate average performance
        if accuracies:
            analytics['avg_accuracy'] = np.mean(accuracies)
            
            # Find best model
            best_idx = np.argmax(accuracies)
            analytics['best_model'] = models[best_idx]
        
        # Recent models (last 5)
        analytics['recent_models'] = models[:5]
        
        return analytics
    
    def create_model_analytics_dashboard(self) -> None:
        """Create visual analytics dashboard"""
        analytics = self.get_model_analytics()
        
        if not analytics:
            st.info("No saved models found for analytics.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", analytics['total_models'])
        with col2:
            st.metric("Total Size", f"{analytics['total_size_mb']:.1f} MB")
        with col3:
            if analytics['avg_accuracy'] > 0:
                st.metric("Avg Performance", f"{analytics['avg_accuracy']:.3f}")
        with col4:
            if analytics['best_model']:
                best_name = analytics['best_model']['model_name']
                st.metric("Best Model", best_name)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Algorithm distribution
            if analytics['algorithms']:
                fig_algo = go.Figure(data=[
                    go.Pie(
                        labels=list(analytics['algorithms'].keys()),
                        values=list(analytics['algorithms'].values()),
                        hole=0.3
                    )
                ])
                fig_algo.update_layout(title="Algorithm Distribution")
                st.plotly_chart(fig_algo, use_container_width=True)
        
        with col2:
            # Problem type distribution
            if analytics['problem_types']:
                fig_problem = go.Figure(data=[
                    go.Pie(
                        labels=list(analytics['problem_types'].keys()),
                        values=list(analytics['problem_types'].values()),
                        hole=0.3
                    )
                ])
                fig_problem.update_layout(title="Problem Type Distribution")
                st.plotly_chart(fig_problem, use_container_width=True)
    
    def compare_models(self, model_filenames: List[str]) -> pd.DataFrame:
        """Compare multiple models side by side"""
        comparison_data = []
        
        for filename in model_filenames:
            data = self.load_model(filename)
            if data and 'metadata' in data:
                metadata = data['metadata']
                
                row = {
                    'Model Name': metadata.get('model_name', 'Unknown'),
                    'Version': metadata.get('version', 'Unknown'),
                    'Algorithm': metadata.get('algorithm', 'Unknown'),
                    'Problem Type': metadata.get('problem_type', 'Unknown'),
                    'Training Time': metadata.get('training_time', 'Unknown'),
                    'File Size (MB)': metadata.get('file_size', 0) / (1024 * 1024)
                }
                
                # Add metrics
                metrics = metadata.get('metrics', {})
                for metric, value in metrics.items():
                    if value is not None:
                        row[metric.replace('_', ' ').title()] = f"{value:.4f}"
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def render_management_tab(self, training_results: Optional[Dict] = None) -> None:
        """Render comprehensive model management interface"""
        st.header("💾 Model Management")
        st.markdown("Save, load, and manage your trained models.")
        
        # Create tabs for different management functions
        mgmt_tabs = st.tabs([
            "💾 Save Models", 
            "📋 Model Library", 
            "📊 Analytics", 
            "⚖️ Compare Models",
            "📤 Export/Import"
        ])
        
        # Save Models Tab
        with mgmt_tabs[0]:
            st.subheader("💾 Save Trained Models")
            
            if not training_results:
                st.warning("⚠️ No trained models in current session. Please train models first.")
            else:
                # Model selection for saving
                available_models = list(training_results.keys())
                selected_model = st.selectbox("Select model to save:", available_models)
                
                if selected_model:
                    model_data = training_results[selected_model]
                    
                    # Model save form
                    with st.form("save_model_form"):
                        st.write(f"**Saving:** {selected_model}")
                        
                        # Display model info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Algorithm:** {model_data['algorithm']}")
                            st.write(f"**Problem Type:** {model_data['problem_type']}")
                        with col2:
                            metrics = model_data['metrics']
                            if 'accuracy' in metrics:
                                st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")
                            elif 'r2_score' in metrics:
                                st.write(f"**R² Score:** {metrics['r2_score']:.4f}")
                        
                        # Save options
                        model_name = st.text_input("Model Name:", value=selected_model.replace(" ", "_"))
                        description = st.text_area("Description:", value="")
                        tags = st.text_input("Tags (comma-separated):", value="")
                        
                        if st.form_submit_button("💾 Save Model", type="primary"):
                            if model_name:
                                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                                filename = self.save_model_with_metadata(
                                    model_data, model_name, description, tag_list
                                )
                                if filename:
                                    st.success(f"✅ Model saved as: {filename}")
                            else:
                                st.error("Please provide a model name.")
        
        # Model Library Tab
        with mgmt_tabs[1]:
            st.subheader("📋 Model Library")
            
            saved_models = self.list_saved_models()
            
            if not saved_models:
                st.info("No saved models found.")
            else:
                # Search and filter
                col1, col2, col3 = st.columns(3)
                with col1:
                    search_term = st.text_input("🔍 Search models:", "")
                with col2:
                    algorithm_filter = st.selectbox(
                        "Filter by algorithm:",
                        ["All"] + list(set(model['algorithm'] for model in saved_models))
                    )
                with col3:
                    problem_type_filter = st.selectbox(
                        "Filter by problem type:",
                        ["All"] + list(set(model['problem_type'] for model in saved_models))
                    )
                
                # Apply filters
                filtered_models = saved_models
                if search_term:
                    filtered_models = [m for m in filtered_models 
                                     if search_term.lower() in m.get('model_name', '').lower()]
                if algorithm_filter != "All":
                    filtered_models = [m for m in filtered_models 
                                     if m.get('algorithm') == algorithm_filter]
                if problem_type_filter != "All":
                    filtered_models = [m for m in filtered_models 
                                     if m.get('problem_type') == problem_type_filter]
                
                # Display models
                for i, model in enumerate(filtered_models):
                    with st.expander(f"📦 {model['model_name']} v{model['version']} - {model['algorithm']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Algorithm:** {model['algorithm']}")
                            st.write(f"**Problem Type:** {model['problem_type']}")
                            st.write(f"**Training Time:** {model['training_time'][:19]}")
                            st.write(f"**File Size:** {model['file_size_mb']:.2f} MB")
                            
                            if model.get('description'):
                                st.write(f"**Description:** {model['description']}")
                            
                            if model.get('tags'):
                                st.write(f"**Tags:** {', '.join(model['tags'])}")
                        
                        with col2:
                            # Display metrics
                            metrics = model.get('metrics', {})
                            st.write("**Performance Metrics:**")
                            for metric, value in metrics.items():
                                if value is not None:
                                    st.write(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
                        
                        # Model actions
                        action_col1, action_col2, action_col3 = st.columns(3)
                        
                        with action_col1:
                            if st.button(f"🔄 Load", key=f"load_{i}"):
                                loaded_data = self.load_model(model['filename'])
                                if loaded_data:
                                    st.session_state.loaded_model = loaded_data
                                    st.success("Model loaded successfully!")
                        
                        with action_col2:
                            if st.button(f"📥 Download", key=f"download_{i}"):
                                model_bytes = self.export_model(model['filename'])
                                if model_bytes:
                                    st.download_button(
                                        "💾 Download",
                                        data=model_bytes,
                                        file_name=model['filename'],
                                        mime="application/octet-stream",
                                        key=f"download_btn_{i}"
                                    )
                        
                        with action_col3:
                            if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                                if self.delete_model(model['filename']):
                                    st.success("Model deleted!")
                                    st.rerun()
        
        # Analytics Tab
        with mgmt_tabs[2]:
            st.subheader("📊 Model Analytics")
            self.create_model_analytics_dashboard()
        
        # Compare Models Tab
        with mgmt_tabs[3]:
            st.subheader("⚖️ Compare Models")
            
            saved_models = self.list_saved_models()
            if len(saved_models) < 2:
                st.info("Need at least 2 saved models for comparison.")
            else:
                model_options = [f"{m['model_name']} v{m['version']}" for m in saved_models]
                selected_for_comparison = st.multiselect(
                    "Select models to compare:",
                    model_options,
                    default=model_options[:2]
                )
                
                if len(selected_for_comparison) >= 2:
                    # Get corresponding filenames
                    selected_filenames = []
                    for selection in selected_for_comparison:
                        for model in saved_models:
                            if f"{model['model_name']} v{model['version']}" == selection:
                                selected_filenames.append(model['filename'])
                                break
                    
                    # Create comparison
                    comparison_df = self.compare_models(selected_filenames)
                    st.dataframe(comparison_df, use_container_width=True)
        
        # Export/Import Tab
        with mgmt_tabs[4]:
            st.subheader("📤 Export/Import Models")
            
            # Export section
            st.markdown("#### 📤 Export Models")
            saved_models = self.list_saved_models()
            
            if saved_models:
                export_model_options = [f"{m['model_name']} v{m['version']}" for m in saved_models]
                selected_export = st.selectbox("Select model to export:", export_model_options)
                
                export_format = st.selectbox(
                    "Export format:",
                    ["joblib", "json_metadata"]
                )
                
                if st.button("📤 Export Model"):
                    # Find the selected model
                    selected_model = None
                    for model in saved_models:
                        if f"{model['model_name']} v{model['version']}" == selected_export:
                            selected_model = model
                            break
                    
                    if selected_model:
                        exported_data = self.export_model(selected_model['filename'], export_format)
                        if exported_data:
                            file_extension = "joblib" if export_format == "joblib" else "json"
                            filename = f"{selected_model['model_name']}_v{selected_model['version']}.{file_extension}"
                            
                            st.download_button(
                                "💾 Download Export",
                                data=exported_data,
                                file_name=filename,
                                mime="application/octet-stream" if export_format == "joblib" else "application/json"
                            )
            
            # Import section
            st.markdown("#### 📥 Import Models")
            uploaded_model = st.file_uploader(
                "Upload model file (.joblib):",
                type=['joblib'],
                help="Upload a previously exported model file"
            )
            
            if uploaded_model:
                if st.button("📥 Import Model"):
                    try:
                        # Save uploaded file
                        import_path = self.models_dir / uploaded_model.name
                        with open(import_path, "wb") as f:
                            f.write(uploaded_model.getbuffer())
                        
                        # Verify the import
                        imported_data = self.load_model(uploaded_model.name)
                        if imported_data:
                            st.success(f"✅ Model imported successfully: {uploaded_model.name}")
                        else:
                            import_path.unlink()  # Remove invalid file
                            st.error("❌ Invalid model file")
                            
                    except Exception as e:
                        st.error(f"Import failed: {str(e)}")
    
    def _get_existing_versions(self, model_name: str) -> List[float]:
        """Get existing version numbers for a model"""
        versions = []
        pattern = f"{model_name}_v*.joblib"
        
        for file in self.models_dir.glob(pattern):
            try:
                version_str = file.stem.split('_v')[1]
                versions.append(float(version_str))
            except (IndexError, ValueError):
                continue
        
        return versions
    
    def _update_metadata_index(self, metadata: Dict, filename: str) -> None:
        """Update global metadata index"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {}
            
            index[filename] = metadata
            
            with open(self.metadata_file, 'w') as f:
                json.dump(index, f, indent=2, default=str)
                
        except Exception as e:
            st.warning(f"Could not update metadata index: {str(e)}")
    
    def _remove_from_metadata_index(self, filename: str) -> None:
        """Remove model from metadata index"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    index = json.load(f)
                
                if filename in index:
                    del index[filename]
                    
                    with open(self.metadata_file, 'w') as f:
                        json.dump(index, f, indent=2, default=str)
                        
        except Exception as e:
            st.warning(f"Could not update metadata index: {str(e)}")
