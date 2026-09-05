import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import pycountry

st.set_page_config(
    page_title="Global EPI Insights",
    page_icon="🌿",
    layout="wide"
)

merged_data = pd.read_csv("data/merged_data.csv")
model = joblib.load("models/rf_pipeline.joblib")
st.sidebar.title("Environmental Performance")
st.sidebar.caption("Socioeconomic Development Dashboard")
page = st.sidebar.radio("Navigation", ["Overview", "Country Explorer", "World Map", "Model Performance", "What-if Simulator"])
if page == "Overview":
    st.header("Overview")
    st.write('This dashboard explores the relationship between socioeconomic indicators and environmental performance across countries.')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Countries", merged_data.shape[0])
    with col2:
        st.metric("Average EPI", round(merged_data['EPI.new'].mean(), 2))
    with col3:
        st.metric("Best Test R²", 0.695)
    st.subheader("Key Findings")

    st.markdown("""
    - Economic prosperity showed the strongest relationship with environmental performance.
    - Random Forest achieved the best predictive performance among the evaluated models.
    - Urbanization was moderately associated with environmental performance and related dimensions.
    - Employment indicators showed little direct association with overall EPI.
    """)

elif page == "Country Explorer":
    st.header("Country Explorer")
    selected_country = st.selectbox("Select a country", merged_data["country"])
    country_data = merged_data[merged_data["country"] == selected_country]

    iso3 = country_data["iso"].iloc[0]
    iso2 = pycountry.countries.get(alpha_3=iso3).alpha_2.lower()
    flag_url = f"https://flagcdn.com/w160/{iso2}.png"
    st.image(flag_url, width=120)

    epi_score = country_data["EPI.new"].iloc[0]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("EPI Score", round(epi_score, 2))
    with col2:
        st.metric("GDP per Capita (US$)", round(country_data["GDP"].iloc[0], 2))
    with col3:
        secondary = country_data["Secondary_enrollment"].iloc[0]
        if pd.isna(secondary):
            secondary = "N/A"
        else:
            secondary = f"{secondary:.2f}%"
        st.metric("Secondary Enrollment, Gross (%)", secondary)
    with col4:
        st.metric("Urban Population (%)", f"{country_data['Urban Population'].iloc[0]:.2f}%")

    log_gdp = np.log(country_data["GDP"].iloc[0])
    country_features = pd.DataFrame({
        "log_GDP": [log_gdp],
        "Secondary_enrollment": [country_data["Secondary_enrollment"].iloc[0]],
        "Urban Population": [country_data["Urban Population"].iloc[0]],
        "Employment_rate": [country_data["Employment_rate"].iloc[0]],
        "Unemployment": [country_data["Unemployment"].iloc[0]]})

    predicted_epi = model.predict(country_features)[0]
    prediction_error = epi_score - predicted_epi
    st.subheader("Model Prediction")
    actual_col, predicted_col, error_col = st.columns(3)

    with actual_col:
        st.metric("Actual EPI", round(epi_score, 2))

    with predicted_col:
        st.metric("Predicted EPI", round(predicted_epi, 2))

    with error_col:
        st.metric("Prediction Error", round(prediction_error, 2))

    st.caption("Prediction Error = Actual EPI - Predicted EPI")

elif page == "World Map":
    st.header("World Map")
    st.write("Explore environmental performance and related indicators across countries.")
    selected_indicator = st.selectbox("Select an environmental indicator",
    ["Environmental Performance", "Air Quality", "Waste Management", "Environmental Health"])
    indicator_columns = {"Environmental Performance" : "EPI.new", "Air Quality" : "AIR.new",  "Waste Management" : "WMG.new",
      "Environmental Health"  : "HLT.new"}
    selected_column = indicator_columns[selected_indicator]

    fig = px.choropleth(
    data_frame=merged_data,
    locations="iso",
    color=selected_column,
    hover_name="country",
    color_continuous_scale="Turbo",
    labels={selected_column: selected_indicator})

    fig.update_geos(
    projection_type="orthographic",
    bgcolor="rgba(0,0,0,0)",
    showocean=True,
    oceancolor="#145A82",
    showland=True,
    landcolor="#1B5E20",
    showcountries=True,
    countrycolor="rgba(255,255,255,0.25)",
    showcoastlines=True,
    coastlinecolor="rgba(255,255,255,0.35)",
    projection_scale=1.00)

    fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=500,
    margin=dict(l=0, r=0, t=20, b=0),
    coloraxis_colorbar=dict(
    title=selected_indicator,
    thickness=18))

    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.header("Model Performance")
    r2_col, mae_col, cv_col = st.columns(3)

    with r2_col:
        st.metric("Test R²", 0.695)

    with mae_col:
        st.metric("Test MAE", 5.53)

    with cv_col:
        st.metric("CV Mean R²", 0.621)

    st.write("**Selected Model:** Random Forest")

    performance_df = pd.DataFrame({"Model" : ["Linear Regression", "Random Forest", 
                                    "Gradient Boosting"], "Test R²" : [0.666, 0.695, 0.673],
                                      "Test MAE" : [6.36, 5.53, 5.69], "CV Mean R²" : [0.577284, 0.621091, 0.575513]})
    st.subheader("Model Comparison")
    fig = px.bar(data_frame = performance_df,x = "Model",y = "Test R²", text = "Test R²", color = "Model",
                color_discrete_map = {
        "Linear Regression": "#5B4B8A",
        "Random Forest": "#9767FF",
        "Gradient Boosting": "#5B4B8A"})
    fig.update_layout(
    height=300,
    margin=dict(t=30, b=20), showlegend=False)
    st.plotly_chart(fig)
    st.dataframe(performance_df)

elif page == "What-if Simulator":
    st.header("What-if Simulator")
    gdp_min = merged_data["GDP"].min()
    gdp_max = merged_data["GDP"].max()
    gdp_default = merged_data["GDP"].median()
    gdp_input = st.slider("GDP per Capita (US$)",min_value = gdp_min, max_value = gdp_max, value = gdp_default)

    secondary_min = merged_data["Secondary_enrollment"].min()
    secondary_max = merged_data["Secondary_enrollment"].max()
    secondary_default = merged_data["Secondary_enrollment"].median()
    secondary_input = st.slider("Secondary Enrollment, Gross (%)", min_value = secondary_min, max_value = secondary_max, value = secondary_default)

    urban_min = merged_data["Urban Population"].min()
    urban_max = merged_data["Urban Population"].max()
    urban_default = merged_data["Urban Population"].median()
    urban_input = st.slider("Urban Population (%)", min_value = urban_min, max_value = urban_max, value = urban_default)

    employment_min = merged_data["Employment_rate"].min()
    employment_max = merged_data["Employment_rate"].max()
    employment_default = merged_data["Employment_rate"].median()
    employment_input = st.slider("Employment Rate (%)", min_value = employment_min, max_value = employment_max, value = employment_default)

    unemployment_min = merged_data["Unemployment"].min()
    unemployment_max = merged_data["Unemployment"].max()
    unemployment_default = merged_data["Unemployment"].median()
    unemployment_input = st.slider("Unemployment Rate (%)", min_value = unemployment_min, max_value = unemployment_max, 
                            value = unemployment_default)
    
    log_gdp_input = np.log(gdp_input)
    scenario_data = pd.DataFrame({
    "log_GDP": [log_gdp_input],
    "Secondary_enrollment": [secondary_input],
    "Urban Population": [urban_input],
    "Employment_rate": [employment_input],
    "Unemployment": [unemployment_input]})

    predicted_epi = model.predict(scenario_data)[0]
    st.metric("Predicted EPI", round(predicted_epi, 2))
    st.caption(
    "Scenario predictions reflect patterns learned from the data and should not be interpreted as causal effects."
)
