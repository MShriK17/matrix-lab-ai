# MatrixLab AI Studio

## Overview
MatrixLab AI Studio is a complete machine learning workflow application built with Streamlit. It provides an end-to-end solution for data analysis, preprocessing, model training, evaluation, and deployment.

**Current Status:** Fully operational and running on Replit

**Technology Stack:**
- Python 3.11
- Streamlit 1.36.0
- scikit-learn 1.5.0
- pandas 2.2.2
- numpy 1.25.2
- Various ML libraries (SHAP, LIME, ydata-profiling)

## Recent Changes
**Date:** November 29, 2025
- Successfully imported GitHub project to Replit
- Fixed dependency conflicts in requirements.txt:
  - Adjusted scipy version to <1.12 (compatibility with ydata-profiling)
  - Adjusted matplotlib to <3.9 (compatibility with ydata-profiling)
  - Adjusted seaborn to <0.13 (compatibility with ydata-profiling)
  - Adjusted numpy to <1.26 (compatibility with ydata-profiling)
  - Adjusted numba to >=0.57.0,<0.59.0 (Python 3.11 compatibility)
  - Adjusted llvmlite to >=0.40.0,<0.42 (compatibility with numba)
  - Adjusted slicer to 0.0.7 (compatibility with shap)
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
- **eda.py**: Exploratory Data Analysis generation
- **preprocessor.py**: Data preprocessing and transformation
- **trainer.py**: Model training functionality
- **predictor.py**: Prediction generation
- **evaluator.py**: Model evaluation and metrics
- **explainability.py**: Model interpretation (SHAP, LIME)
- **manager.py**: Model management and versioning
- **visualization.py**: Data visualization utilities

### Utilities
- **utils/session_utils.py**: Session state management

### Models
- **models/**: Stores trained models and metadata
  - Pre-loaded: Linear_Regression_v1.0.joblib (R² score: 0.999)

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
- CORS: disabled (required for Replit)
- XSRF protection: disabled (required for Replit proxy)

### Deployment
- Type: Autoscale (stateless web application)
- Command: `streamlit run Main_App.py --server.port=5000 --server.address=0.0.0.0`
- Port: 5000 (automatically exposed for web preview)

## Development

### Running Locally
The application automatically starts via the configured workflow. To manually run:
```bash
streamlit run Main_App.py
```

### Dependencies
All dependencies are listed in `requirements.txt`. Install with:
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
- The EDA module uses ydata-profiling for comprehensive reports
- Explainability features use SHAP and LIME libraries
- All visualizations are interactive using Plotly

## Known Issues
- LSP shows import errors for streamlit, pandas, numpy - these are false positives and can be ignored
- Some "possibly unbound" warnings in Main_App.py are also false positives from the type checker

## Future Enhancements
*To be discussed with user as project evolves*
