# Automated Data Analyzer + AutoML (LazyPredict)

## Overview
This is a Streamlit-based web application that provides an automated data analysis and machine learning solution. It allows users to upload a CSV file, visualize data, and automatically compare various machine learning models (classification or regression) using LazyPredict.

## Features
- **CSV File Upload**: Easily upload your data in CSV format.
- **Data Preview & Overview**: Get a quick look at your data, its shape, and column types.
- **Missing Value Analysis**: Identify and summarize missing values in your dataset.
- **Descriptive Statistics**: View comprehensive descriptive statistics for all columns.
- **Automated Exploratory Data Analysis (EDA)**:
  - **Numerical Analysis**: Visualize distributions of numerical columns using histograms.
  - **Categorical Analysis**: Display count plots for categorical columns.
  - **Correlation Heatmap**: (For numerical columns) Visualize correlations between numerical features.
- **Automated Machine Learning (AutoML) with LazyPredict**:
  - **Target Column Selection**: Select your target variable for predictive modeling.
  - **Automatic Problem Detection**: Automatically detects whether the problem is classification or regression based on the target column.
  - **Model Comparison**: Rapidly builds and compares numerous machine learning models, providing performance metrics.
  - **Prediction Preview**: Shows a preview of actual vs. predicted values from a simple RandomForest model.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/sanniv0/Automated-Data-Analyzer.git
   cd Automated-Data-Analyzer
   ```
2. Install dependencies. It is highly recommended to create a virtual environment first:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate # On Windows
   # source venv/bin/activate # On macOS/Linux
   pip install -r requirements.txt
   ```
   *Note: If you encounter issues with `pycaret` or `numpy` installation related to GCC, ensure your GCC compiler is updated to version 8.4 or newer. On Windows, consider using MSYS2 for managing GCC.*

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Configuration
No specific configuration files are required. The application runs directly from `app.py`.

## Usage
1. Run the Streamlit application as described in the Installation section.
2. Upload your CSV file using the file uploader on the sidebar.
3. Explore the data previews, overviews, and visualizations.
4. In the AutoML section, select your target column from the dropdown.
5. The application will automatically detect the problem type (classification/regression) and display a comparison of various models.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT