"""
Session Management Utilities for MatrixLab AI Studio
Handle session state, workflow persistence, and project management
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import datetime
import warnings
warnings.filterwarnings('ignore')

class SessionManager:
    """Comprehensive session state management"""
    
    def __init__(self):
        self.session_file = Path("session_data.json")
        self.projects_dir = Path("projects")
        self.projects_dir.mkdir(exist_ok=True)
        
        # Initialize default session state
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state with default values"""
        defaults = {
            'current_step': 0,
            'project_name': 'New Project',
            'data_uploaded': False,
            'preprocessing_applied': False,
            'workflow_completed': False,
            'project_created_at': datetime.datetime.now().isoformat(),
            'last_activity': datetime.datetime.now().isoformat(),
            'workflow_history': [],
            'user_preferences': {
                'theme': 'light',
                'auto_save': True,
                'show_tips': True
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def save_session_state(self, project_name: Optional[str] = None) -> bool:
        """Save current session state to file"""
        try:
            project_name = project_name or st.session_state.get('project_name', 'untitled')
            project_name = self._sanitize_filename(project_name)
            
            # Prepare session data
            session_data = {}
            
            # Core workflow data
            serializable_keys = [
                'current_step', 'project_name', 'data_uploaded', 'preprocessing_applied',
                'workflow_completed', 'project_created_at', 'last_activity', 
                'workflow_history', 'user_preferences'
            ]
            
            for key in serializable_keys:
                if key in st.session_state:
                    session_data[key] = st.session_state[key]
            
            # Handle data separately (convert to JSON serializable format)
            if 'data' in st.session_state:
                df = st.session_state.data
                session_data['data_info'] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.to_dict(),
                    'sample_data': df.head(5).to_dict('records')
                }
            
            # Handle preprocessing info
            if 'preprocessing_info' in st.session_state:
                preprocessing_info = st.session_state.preprocessing_info.copy()
                # Remove non-serializable objects
                if 'label_encoders' in preprocessing_info:
                    preprocessing_info['label_encoders_classes'] = {
                        k: v.classes_.tolist() if hasattr(v, 'classes_') else []
                        for k, v in preprocessing_info['label_encoders'].items()
                    }
                    del preprocessing_info['label_encoders']
                
                if 'target_encoder' in preprocessing_info and preprocessing_info['target_encoder']:
                    if hasattr(preprocessing_info['target_encoder'], 'classes_'):
                        preprocessing_info['target_encoder_classes'] = preprocessing_info['target_encoder'].classes_.tolist()
                    del preprocessing_info['target_encoder']
                
                session_data['preprocessing_info'] = preprocessing_info
            
            # Handle training results metadata
            if 'training_results' in st.session_state:
                training_metadata = {}
                for model_name, result in st.session_state.training_results.items():
                    training_metadata[model_name] = {
                        'algorithm': result.get('algorithm'),
                        'problem_type': result.get('problem_type'),
                        'metrics': result.get('metrics'),
                        'training_time': result.get('training_time').isoformat() if result.get('training_time') else None,
                        'best_params': result.get('best_params'),
                        'cv_scores': result.get('cv_scores').tolist() if result.get('cv_scores') is not None else None
                    }
                session_data['training_metadata'] = training_metadata
            
            # Update last activity
            session_data['last_activity'] = datetime.datetime.now().isoformat()
            
            # Save to project file
            project_file = self.projects_dir / f"{project_name}_session.json"
            with open(project_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving session: {str(e)}")
            return False
    
    def load_session_state(self, project_name: str) -> bool:
        """Load session state from file"""
        try:
            project_name = self._sanitize_filename(project_name)
            project_file = self.projects_dir / f"{project_name}_session.json"
            
            if not project_file.exists():
                st.error(f"Project file not found: {project_name}")
                return False
            
            with open(project_file, 'r') as f:
                session_data = json.load(f)
            
            # Restore session state
            for key, value in session_data.items():
                if key not in ['data_info', 'training_metadata']:
                    st.session_state[key] = value
            
            # Handle data info (user will need to re-upload data)
            if 'data_info' in session_data:
                st.session_state.data_info = session_data['data_info']
                st.info("Project loaded. Please re-upload your data to continue the workflow.")
            
            # Handle training metadata
            if 'training_metadata' in session_data:
                st.session_state.training_metadata = session_data['training_metadata']
            
            st.success(f"Project '{project_name}' loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error loading session: {str(e)}")
            return False
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all saved projects"""
        projects = []
        
        for project_file in self.projects_dir.glob("*_session.json"):
            try:
                with open(project_file, 'r') as f:
                    session_data = json.load(f)
                
                project_info = {
                    'name': session_data.get('project_name', 'Untitled'),
                    'file_name': project_file.stem.replace('_session', ''),
                    'created_at': session_data.get('project_created_at'),
                    'last_activity': session_data.get('last_activity'),
                    'current_step': session_data.get('current_step', 0),
                    'workflow_completed': session_data.get('workflow_completed', False),
                    'data_uploaded': session_data.get('data_uploaded', False),
                    'file_path': str(project_file)
                }
                projects.append(project_info)
                
            except Exception as e:
                st.warning(f"Could not load project info from {project_file.name}: {str(e)}")
        
        # Sort by last activity
        projects.sort(key=lambda x: x.get('last_activity', ''), reverse=True)
        return projects
    
    def delete_project(self, project_name: str) -> bool:
        """Delete a saved project"""
        try:
            project_name = self._sanitize_filename(project_name)
            project_file = self.projects_dir / f"{project_name}_session.json"
            
            if project_file.exists():
                project_file.unlink()
                return True
            return False
            
        except Exception as e:
            st.error(f"Error deleting project: {str(e)}")
            return False
    
    def export_project(self, project_name: str) -> Optional[bytes]:
        """Export project as JSON file"""
        try:
            project_name = self._sanitize_filename(project_name)
            project_file = self.projects_dir / f"{project_name}_session.json"
            
            if project_file.exists():
                return project_file.read_bytes()
            return None
            
        except Exception as e:
            st.error(f"Error exporting project: {str(e)}")
            return None
    
    def import_project(self, uploaded_file) -> bool:
        """Import project from uploaded JSON file"""
        try:
            # Read and validate the uploaded file
            project_data = json.load(uploaded_file)
            
            # Validate required fields
            required_fields = ['project_name', 'project_created_at']
            for field in required_fields:
                if field not in project_data:
                    st.error(f"Invalid project file: missing {field}")
                    return False
            
            # Generate unique filename if project already exists
            project_name = self._sanitize_filename(project_data['project_name'])
            project_file = self.projects_dir / f"{project_name}_session.json"
            
            counter = 1
            while project_file.exists():
                new_name = f"{project_name}_{counter}"
                project_file = self.projects_dir / f"{new_name}_session.json"
                counter += 1
            
            # Save imported project
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            st.error(f"Error importing project: {str(e)}")
            return False
    
    def track_workflow_step(self, step_name: str, details: Optional[Dict] = None) -> None:
        """Track workflow step completion"""
        if 'workflow_history' not in st.session_state:
            st.session_state.workflow_history = []
        
        step_record = {
            'step_name': step_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'details': details or {}
        }
        
        st.session_state.workflow_history.append(step_record)
        st.session_state.last_activity = datetime.datetime.now().isoformat()
        
        # Auto-save if enabled
        if st.session_state.get('user_preferences', {}).get('auto_save', True):
            self.save_session_state()
    
    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get current workflow progress"""
        total_steps = 8
        current_step = st.session_state.get('current_step', 0)
        completed_steps = len(st.session_state.get('workflow_history', []))
        
        progress = {
            'current_step': current_step,
            'total_steps': total_steps,
            'progress_percentage': (current_step / total_steps) * 100,
            'completed_steps': completed_steps,
            'workflow_history': st.session_state.get('workflow_history', [])
        }
        
        return progress
    
    def reset_session(self, keep_preferences: bool = True) -> None:
        """Reset session state for new project"""
        preferences = st.session_state.get('user_preferences', {}) if keep_preferences else {}
        
        # Clear all session state except preferences
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            if key != 'user_preferences' or not keep_preferences:
                del st.session_state[key]
        
        # Reinitialize
        self._initialize_session_state()
        
        if keep_preferences:
            st.session_state.user_preferences = preferences
    
    def create_project_summary(self) -> Dict[str, Any]:
        """Create comprehensive project summary"""
        summary = {
            'project_info': {
                'name': st.session_state.get('project_name', 'Untitled'),
                'created_at': st.session_state.get('project_created_at'),
                'last_activity': st.session_state.get('last_activity'),
                'current_step': st.session_state.get('current_step', 0)
            },
            'data_info': {},
            'preprocessing_info': {},
            'training_info': {},
            'workflow_progress': self.get_workflow_progress()
        }
        
        # Data information
        if 'data' in st.session_state:
            df = st.session_state.data
            summary['data_info'] = {
                'shape': df.shape,
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        
        # Preprocessing information
        if 'preprocessing_info' in st.session_state:
            preprocessing = st.session_state.preprocessing_info
            summary['preprocessing_info'] = {
                'feature_count': len(preprocessing.get('feature_names', [])),
                'target_column': preprocessing.get('target_column'),
                'encoders_used': len(preprocessing.get('label_encoders', {}))
            }
        
        # Training information
        if 'training_results' in st.session_state:
            training = st.session_state.training_results
            summary['training_info'] = {
                'models_trained': len(training),
                'algorithms': list(training.keys()),
                'best_model': self._find_best_model(training)
            }
        
        return summary
    
    def render_session_management_ui(self) -> None:
        """Render session management interface"""
        st.sidebar.markdown("## 💾 Project Management")
        
        # Current project info
        current_project = st.session_state.get('project_name', 'New Project')
        st.sidebar.markdown(f"**Current Project:** {current_project}")
        
        # Progress indicator
        progress = self.get_workflow_progress()
        st.sidebar.progress(progress['progress_percentage'] / 100)
        st.sidebar.markdown(f"Step {progress['current_step'] + 1}/{progress['total_steps']}")
        
        # Project actions
        with st.sidebar.expander("🔧 Project Actions"):
            # Save project
            if st.button("💾 Save Project"):
                if self.save_session_state():
                    st.success("Project saved!")
            
            # New project
            if st.button("📝 New Project"):
                if st.session_state.get('data_uploaded', False):
                    if st.button("⚠️ Confirm - Start New Project"):
                        self.reset_session()
                        st.rerun()
                else:
                    self.reset_session()
                    st.rerun()
        
        # Load existing project
        with st.sidebar.expander("📂 Load Project"):
            projects = self.list_projects()
            
            if projects:
                project_options = [f"{p['name']} ({p['last_activity'][:10]})" for p in projects]
                selected_project = st.selectbox("Select project:", project_options)
                
                if st.button("📂 Load Selected"):
                    project_name = projects[project_options.index(selected_project)]['file_name']
                    if self.load_session_state(project_name):
                        st.rerun()
            else:
                st.info("No saved projects found")
        
        # Project settings
        with st.sidebar.expander("⚙️ Settings"):
            # Auto-save toggle
            auto_save = st.checkbox(
                "Auto-save",
                value=st.session_state.get('user_preferences', {}).get('auto_save', True)
            )
            
            # Show tips toggle
            show_tips = st.checkbox(
                "Show tips",
                value=st.session_state.get('user_preferences', {}).get('show_tips', True)
            )
            
            # Update preferences
            if 'user_preferences' not in st.session_state:
                st.session_state.user_preferences = {}
            
            st.session_state.user_preferences['auto_save'] = auto_save
            st.session_state.user_preferences['show_tips'] = show_tips
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file operations"""
        import re
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = filename.strip()
        return filename if filename else 'untitled'
    
    def _find_best_model(self, training_results: Dict) -> Optional[str]:
        """Find the best performing model"""
        if not training_results:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, result in training_results.items():
            metrics = result.get('metrics', {})
            
            # Use accuracy for classification, r2_score for regression
            score = metrics.get('accuracy') or metrics.get('r2_score') or 0
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
