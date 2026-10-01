import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Road Accident Severity Analysis",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Road Accident Severity Analysis")
st.caption("Explore accident patterns and predict accident severity")

# Load dataset and trained model
@st.cache_data
def load_data():
    data = pd.read_csv("road_accident.csv.csv")
    data["accident_date"] = pd.to_datetime(
        data["accident_date"], errors="coerce"
    )
    data["fatal_binary"] = (
        data["accident_severity"] == "Fatal"
    ).astype(int)
    return data

@st.cache_resource
def load_model():
    return joblib.load("road_accident_model.joblib")

df = load_data()
model = load_model()

# Dashboard navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Accident Analysis", "Severity Prediction"]
)

if page == "Overview":
    st.header("Accident Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Accidents", f"{len(df):,}")
    c2.metric("Fatal Accidents", f"{df['fatal_binary'].sum():,}")
    c3.metric(
        "Non-Fatal Accidents",
        f"{(df['fatal_binary'] == 0).sum():,}"
    )

    st.subheader("Accidents by Severity")
    st.bar_chart(df["accident_severity"].value_counts())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Accident Analysis":
    st.header("Explore Accident Risk Factors")

    factor = st.selectbox(
        "Choose a factor",
        [
            "alcohol_involved",
            "lighting_condition",
            "weather_condition",
            "road_condition",
            "accident_cause",
            "vehicle_type",
            "road_type",
            "traffic_density"
        ]
    )

    fatal_rates = (
        df.groupby(factor)["fatal_binary"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.subheader("Fatal Accident Percentage")
    st.bar_chart(fatal_rates)
    st.caption(
        "Percentages describe this synthetic dataset and do not "
        "establish real-world causation."
    )

elif page == "Severity Prediction":
    st.header("Predict Accident Severity")
    st.info(
        "Enter the accident details below. The prediction is based "
        "on the trained model and is not a real-world safety assessment."
    )

    # Recreate the same input features used during training
    features = df.copy()
    features["accident_month"] = features["accident_date"].dt.month
    features["accident_hour"] = pd.to_datetime(
        features["accident_time"],
        format="%H:%M:%S",
        errors="coerce"
    ).dt.hour

    X = features.drop(
        columns=[
            "fatal_binary",
            "accident_severity",
            "number_of_fatalities",
            "number_of_injuries",
            "accident_id",
            "accident_date",
            "accident_time"
        ],
        errors="ignore"
    )

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    with st.form("prediction_form"):
        st.subheader("Accident Details")
        values = {}

        for col in categorical_cols:
            options = sorted(X[col].dropna().astype(str).unique())
            if options:
                values[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options
                )

        for col in numerical_cols:
            median_value = float(X[col].median())
            values[col] = st.number_input(
                col.replace("_", " ").title(),
                value=median_value
            )

        submitted = st.form_submit_button("Predict Severity")

    if submitted:
        input_df = pd.DataFrame(
            [[values[col] for col in X.columns]],
            columns=X.columns
        )

        prediction = int(model.predict(input_df)[0])

        if prediction == 1:
            st.error("Model Prediction: Fatal")
        else:
            st.success("Model Prediction: Non-Fatal")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.write(
                f"Model-estimated fatal probability: "
                f"{probabilities[1] * 100:.2f}%"
            )
            st.caption(
                "This probability reflects the trained synthetic-data "
                "model and is not a validated real-world risk estimate."
            )
