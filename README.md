# Environmental Performance Analysis

A cross-country data science project exploring the relationship between socioeconomic development and environmental performance using Yale EPI and World Bank data.

The project combines exploratory data analysis, statistical analysis, machine learning, predictive modeling, and an interactive Streamlit dashboard.

## Project Overview

This project investigates how socioeconomic development is associated with environmental performance across countries.

Using environmental indicators from the Yale Environmental Performance Index (EPI) and socioeconomic indicators from the World Bank, the analysis examines relationships involving economic prosperity, education, urbanization, and employment.

In addition to exploratory analysis, multiple regression models are evaluated to predict countries' environmental performance based on socioeconomic indicators.

## Research Questions

The analysis is guided by the following questions:

1. Is economic prosperity associated with environmental performance?
2. Do countries with higher educational participation show higher environmental performance?
3. How is urbanization associated with environmental performance, and does this relationship differ across environmental dimensions such as air quality, waste management, and environmental health?
4. Are employment and unemployment rates associated with environmental performance?
5. Which countries perform substantially better or worse environmentally than their economic prosperity might suggest?
6. Which socioeconomic indicators show the strongest relationships with environmental performance?

### Predictive Questions

1. Can socioeconomic indicators be used to predict a country's environmental performance?
2. How do different regression models compare in predicting environmental performance?

## Data Sources

The project combines data from two primary sources:

- **Yale Environmental Performance Index (EPI):** Environmental performance indicators, including the overall EPI score, air quality, waste management, and environmental health.
- **World Bank:** Socioeconomic indicators including GDP per capita, secondary school enrollment, urban population, employment rate, and unemployment rate.

The datasets were cleaned, transformed, and merged at the country level to create the final dataset used for analysis and modeling.

## Key Findings

- Economic prosperity showed the strongest relationship with environmental performance, with log GDP per capita having a correlation of approximately **0.78** with EPI.
- Secondary school enrollment and urbanization also showed positive relationships with environmental performance.
- Urbanization was moderately associated with air quality, waste management, and environmental health.
- Employment and unemployment rates showed very weak direct relationships with overall EPI.
- Some countries performed substantially better or worse environmentally than expected based on their economic prosperity alone.
- Among the evaluated models, **Random Forest** achieved the best predictive performance.

## Machine Learning

The following regression models were evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Missing feature values were handled using median imputation. Model performance was evaluated using a train-test split and cross-validation.

| Model | Test R² | Test MAE | CV Mean R² |
|---|---:|---:|---:|
| Linear Regression | 0.666 | 6.36 | 0.577 |
| Random Forest | **0.695** | **5.53** | **0.621** |
| Gradient Boosting | 0.673 | 5.69 | 0.576 |

**Random Forest** was selected as the final model because it achieved the strongest overall predictive performance among the evaluated models.

## Interactive Dashboard

An interactive Streamlit dashboard was developed to make the analysis and predictive model easier to explore.

The dashboard includes:

- **Overview:** Summary metrics and key findings from the analysis.
- **Country Explorer:** Country-level socioeconomic and environmental indicators alongside actual and predicted EPI scores.
- **World Map:** An interactive global visualization of EPI, air quality, waste management, and environmental health.
- **Model Performance:** Comparison of the evaluated regression models and their performance metrics.
- **What-if Simulator:** Allows users to modify socioeconomic indicators and explore the EPI predictions produced by the trained model.

> The simulator reflects patterns learned from the dataset and should not be interpreted as estimating causal effects.

## Project Structure

```text
environmental-performance-analysis/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── epi/
│   ├── raw_indicators/
│   ├── world_bank/
│   └── merged_data.csv
├── models/
│   └── rf_pipeline.joblib
├── app.py
├── environmental_analysis.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

- **environmental_analysis.ipynb:** Data preparation, exploratory analysis, statistical analysis, and machine learning workflow.
- **app.py:** Streamlit dashboard application.
- **data/:** Environmental and socioeconomic datasets used in the project.
- **models/:** Exported machine learning pipeline used by the dashboard.
- **requirements.txt:** Python dependencies required to run the application.
- **.streamlit/config.toml:** Streamlit theme configuration.

## Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Adapala4/environmental-performance-analysis.git
```

2. Navigate to the project directory:

```bash
cd environmental-performance-analysis
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Limitations

- The analysis is based on cross-sectional country-level data and should not be interpreted as evidence of causal relationships.
- Missing values in the socioeconomic indicators may limit the amount of information available for some countries.
- The dataset is relatively small for machine learning, with 176 countries included in the analysis.
- Model predictions reflect patterns present in the available data and may not generalize to different time periods or unseen conditions.