import streamlit as st
import pandas as pd
import joblib
import os
from data_preprocessing import load_and_clean, feature_engineer

# Paths (update if the saved model name changed)
model_path = 'salarysense_lightgbm.pkl'
data_path = 'adult 3.csv'

# Verify model
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please run model_training.py first.")
    st.stop()

# Load model
model = joblib.load(model_path)

# Load raw data for defaults
raw_df = load_and_clean(data_path)
fnlwgt_mean = raw_df['fnlwgt'].mean()

# Extract categorical options
cat_transformer = model.named_steps['pre'].named_transformers_['cat']
(workclass_opts, education_opts, marital_opts, occ_opts,
 rel_opts, race_opts, gen_opts, country_opts, agebin_opts) = cat_transformer.categories_

# UI setup
st.set_page_config(page_title="SalarySense - Prediction", layout="centered")
st.title("💼 SalarySense: Employee Income Prediction")

# Sidebar inputs
st.sidebar.header("Input Features")
age = st.sidebar.slider('Age', 17, 90, 30)
workclass = st.sidebar.selectbox('Workclass', sorted(workclass_opts))
education = st.sidebar.selectbox('Education', sorted(education_opts))
marital = st.sidebar.selectbox('Marital Status', sorted(marital_opts))
occupation = st.sidebar.selectbox('Occupation', sorted(occ_opts))
relationship = st.sidebar.selectbox('Relationship', sorted(rel_opts))
race = st.sidebar.selectbox('Race', sorted(race_opts))
gender = st.sidebar.selectbox('Gender', sorted(gen_opts))
native_country = st.sidebar.selectbox('Native Country', sorted(country_opts))
capital_gain = st.sidebar.number_input('Capital Gain', min_value=0, value=0)
capital_loss = st.sidebar.number_input('Capital Loss', min_value=0, value=0)
hours = st.sidebar.slider('Hours per Week', 1, 99, 40)

# Prepare DataFrame
inp = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt_mean,
    'education': education,
    'marital-status': marital,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours,
    'native-country': native_country
}
input_df = pd.DataFrame([inp])
input_df = feature_engineer(input_df)

# Predict button
if st.sidebar.button('Predict'):
    pred = model.predict(input_df)[0]
    res = '>50K' if pred==1 else '<=50K'
    st.subheader("Prediction Result")
    st.write(f"### Income: **{res}**")

# Batch CSV
st.sidebar.header("Batch Prediction")
file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if file:
    batch = pd.read_csv(file)
    batch = feature_engineer(batch)
    preds = model.predict(batch)
    out = pd.DataFrame({'prediction': ['>50K' if p==1 else '<=50K' for p in preds]})
    st.write(out)