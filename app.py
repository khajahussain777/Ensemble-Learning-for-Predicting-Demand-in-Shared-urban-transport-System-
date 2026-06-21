"""Streamlit app for the Ensemble Learning demand predictor demo."""
import os
import streamlit as st
import pandas as pd
from models.train_model import (
    generate_synthetic_data,
    train_and_save_model,
    load_model,
    predict_from_input,
    DEFAULT_MODEL_PATH,
    load_dataset_from_csv,
)


def main():
    st.title("Ensemble Learning — Demand Prediction (Shared Urban Transport)")

    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=["csv"])
    uploaded_df = None
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV uploaded")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")

    action = st.sidebar.selectbox("Action", ["Show sample data", "Train model", "Load model", "Predict"])

    if action == "Show sample data":
        if uploaded_df is not None:
            st.write("Preview of uploaded dataset")
            st.dataframe(uploaded_df.head(200))
            st.write(uploaded_df.describe())
        else:
            n = st.sidebar.slider("Rows", 100, 10000, 500)
            df = generate_synthetic_data(n)
            st.write("Sample of generated data")
            st.dataframe(df.head(200))
            st.write(df.describe())

    elif action == "Train model":
        use_uploaded = uploaded_df is not None and st.sidebar.checkbox("Use uploaded dataset for training", value=True)
        if not use_uploaded:
            n = st.sidebar.slider("Samples", 500, 20000, 2000)

        if st.sidebar.button("Start training"):
            with st.spinner("Training model — this may take a while"):
                if use_uploaded:
                    # if user uploaded a file, save to temp and load via helper to validate
                    try:
                        temp_path = os.path.join("data", "uploaded.csv")
                        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                        uploaded_df.to_csv(temp_path, index=False)
                        df = load_dataset_from_csv(temp_path)
                    except Exception as e:
                        st.error(f"Failed to use uploaded dataset: {e}")
                        return
                else:
                    df = generate_synthetic_data(n)
                model, metrics = train_and_save_model(df, model_path=DEFAULT_MODEL_PATH)
            st.success("Training finished")
            st.write(metrics)

    elif action == "Load model":
        model = load_model(DEFAULT_MODEL_PATH)
        if model is None:
            st.warning("No model found. Train a model first.")
        else:
            st.success("Model loaded successfully")

    elif action == "Predict":
        model = load_model(DEFAULT_MODEL_PATH)
        if model is None:
            st.warning("No model found. Train a model first.")
            return

        st.sidebar.write("Input features for a single prediction")
        hour = st.sidebar.slider("Hour of day", 0, 23, 8)
        day_of_week = st.sidebar.selectbox("Day of week", list(range(7)), index=0)
        is_weekend = st.sidebar.checkbox("Is weekend", value=False)
        weather = st.sidebar.slider("Weather (0 good -> 1 bad)", 0.0, 1.0, 0.1)
        temperature = st.sidebar.slider("Temperature (C)", -20.0, 40.0, 18.0)
        stations_nearby = st.sidebar.slider("Stations nearby", 0, 100, 5)
        population_density = st.sidebar.slider("Population density", 100, 30000, 5000)
        promotion = st.sidebar.checkbox("Promotion running", value=False)
        prev_demand = st.sidebar.slider("Previous hour demand", 0.0, 2000.0, 50.0)

        input_df = pd.DataFrame([
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": int(is_weekend),
                "weather": weather,
                "temperature": temperature,
                "stations_nearby": stations_nearby,
                "population_density": population_density,
                "promotion": int(promotion),
                "prev_demand": prev_demand,
            }
        ])

        if st.sidebar.button("Predict"):
            pred = predict_from_input(model, input_df)
            st.metric(label="Predicted demand", value=round(float(pred[0]), 2))


if __name__ == "__main__":
    main()
