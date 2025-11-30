# MatrixLab AI Studio

## Overview
MatrixLab AI Studio is a complete machine learning workflow application built with Streamlit. It provides an end-to-end solution for data analysis, preprocessing, model training, evaluation, and deployment.

**Current Status:** Fully operational and ready for Streamlit Cloud deployment

**Technology Stack:**
- Python 3.11
- Streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- SHAP and LIME for model explainability
- Plotly for interactive visualizations

## Recent Changes
**Date:** November 30, 2025
- Cleaned up requirements.txt for Streamlit Cloud compatibility
- Removed ydata-profiling (caused dependency conflicts) - EDA now uses built-in fallback
- Simplified dependencies to minimal set, letting pip resolve sub-dependencies
- Application now deploys without errors on Streamlit Cloud

**Date:** November 29, 2025
- Successfully imported GitHub project to Replit
- Created Streamlit configuration for Replit environment
- Set up workflow to run on port 5000
- Configured deployment settings for autoscale deployment
- Added .gitignore for Python project

## Project Architecture

### Main Application
- **Main_App.py**: Central application file with 8-step ML workflow

### Modules
Located in the `modules/` directory:
- **data_loader.py**: Handles data upload and validation
- **eda.py**: Exploratory Data Analysis generation (with fallback when ydata-profiling unavailable)
- **preprocessor.py**: Data preprocessing and transformation
- **trainer.py**: Model training functionality
- **predictor.py**: Prediction generation
- **evaluator.py**: Model evaluation and metrics
- **explainability.py**: Model interpretation (SHAP, LIME) - gracefully handles missing libraries
- **manager.py**: Model management and versioning
- **visualization.py**: Data visualization utilities

### Utilities
- **utils/session_utils.py**: Session state management

### Models
- **models/**: Stores trained models and metadata
  - Pre-loaded: Linear_Regression_v1.0.joblib (R2 score: 0.999)

## Workflow Steps

1. **Data Upload & Visualization**: Upload CSV, Excel, or JSON datasets
2. **Data Exploration**: Generate comprehensive EDA reports
3. **Data Preprocessing**: Handle missing values, feature selection
4. **Model Training**: Train various ML models
5. **Model Evaluation**: Assess model performance
6. **Predictions**: Generate predictions on new data
7. **Model Explainability**: Understand model decisions
8. **Model Management**: Save, load, and version models

## Configuration

### Streamlit Settings
Located in `.streamlit/config.toml`:
- Server port: 5000
- Address: 0.0.0.0
- CORS: disabled (required for Replit/Cloud hosting)
- XSRF protection: disabled (required for proxy environments)

### Deployment
**Streamlit Cloud:**
- Connect your GitHub repository
- Main file path: `Main_App.py`
- Python version: 3.11

**Replit:**
- Type: Autoscale (stateless web application)
- Command: `streamlit run Main_App.py --server.port=5000 --server.address=0.0.0.0`
- Port: 5000

## Development

### Running Locally
The application automatically starts via the configured workflow. To manually run:
```bash
streamlit run Main_App.py
```

### Dependencies
Minimal, clean dependencies in `requirements.txt`:
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0,<2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
shap>=0.42.0
lime>=0.2.0.1
tqdm>=4.65.0
Pillow>=9.5.0
openpyxl>=3.1.0
```

Install with:
```bash
pip install -r requirements.txt
```

### Adding New Features
When adding new features:
1. Create new modules in the `modules/` directory
2. Import and initialize in Main_App.py
3. Add to the appropriate workflow step
4. Update this documentation

## User Preferences
*No specific user preferences recorded yet*

## Notes
- The application uses Streamlit's session state for maintaining workflow progress
- Models are persisted in the `models/` directory with metadata
- The EDA module has a built-in fallback when ydata-profiling is not available
- Explainability features gracefully handle missing SHAP/LIME libraries
- All visualizations are interactive using Plotly

## Known Issues
- LSP shows import errors for streamlit, pandas, numpy - these are false positives and can be ignored

## Future Enhancements
*To be discussed with user as project evolves*
