import streamlit as st
import pandas as pd
import joblib


# Load the trained model and column names
model = joblib.load("../models/rf_model.pkl")
model_columns = joblib.load("../models/model_columns.pkl")


# Page settings
st.set_page_config(
    page_title="SRHR Message Listening Prediction",
    page_icon="📢",
    layout="centered"
)


# Title
st.title("📢 SRHR Message Listening Prediction")

st.write(
    "This application predicts whether a pupil is likely "
    "to have listened to SRHR messages."
)

st.divider()


# -------------------------
# PUPIL INFORMATION
# -------------------------

st.subheader("👤 Pupil Information")

age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=18
)

age_group = st.selectbox(
    "Age Group",
    ["15-19yrs", "20-24yrs", "25+yrs"]
)

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

district = st.selectbox(
    "District",
    [
        "Hoima",
        "Jinja",
        "Kampala",
        "Mbale",
        "Mbarara",
        "Mukono",
        "Napak"
    ]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Student/Pupil",
        "Teacher",
        "Farmer (peasant farmer)",
        "Business man/woman",
        "Civil servant",
        "Bodaboda",
        "House girl",
        "Housewife",
        "Others"
    ]
)


# -------------------------
# SERVICE INFORMATION
# -------------------------

st.divider()

st.subheader("🏥 Service Experience")

service_comprehensive = st.selectbox(
    "Was the service comprehensive?",
    ["Yes", "No"]
)

health_worker_knowledgeable = st.selectbox(
    "Was the health worker knowledgeable?",
    ["Yes", "No"]
)

health_worker_caring = st.selectbox(
    "Was the health worker caring?",
    ["Yes", "No"]
)

money_spent = st.number_input(
    "Money spent on health (UGX)",
    min_value=0,
    value=0
)

satisfied = st.selectbox(
    "Were you satisfied with the service?",
    ["Yes", "No"]
)

number_messages = st.number_input(
    "Number of messages recalled",
    min_value=0,
    max_value=20,
    value=0
)


# -------------------------
# PREDICTION BUTTON
# -------------------------

st.divider()

if st.button("🔮 Predict", use_container_width=True):

    # Create one empty row with exactly
    # the same columns used by the model
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_columns
    )

    # Numerical variables
    if "Age" in input_data.columns:
        input_data["Age"] = age

    if "No of messages recalled" in input_data.columns:
        input_data["No of messages recalled"] = number_messages

    if "Money(UGX) spent on health" in input_data.columns:
        input_data["Money(UGX) spent on health"] = money_spent


    # Service variables
    if "Service comprehesive" in input_data.columns:
        input_data["Service comprehesive"] = (
            1 if service_comprehensive == "Yes" else 0
        )

    if "H/w knowlegeable" in input_data.columns:
        input_data["H/w knowlegeable"] = (
            1 if health_worker_knowledgeable == "Yes" else 0
        )

    if "H/w caring" in input_data.columns:
        input_data["H/w caring"] = (
            1 if health_worker_caring == "Yes" else 0
        )

    if "Satisfied with service" in input_data.columns:
        input_data["Satisfied with service"] = (
            1 if satisfied == "Yes" else 0
        )


    # Age group
    age_column = "age group_" + age_group

    if age_column in input_data.columns:
        input_data[age_column] = 1


    # Sex
    sex_column = "Sex_" + sex

    if sex_column in input_data.columns:
        input_data[sex_column] = 1


    # District
    district_column = "District_" + district

    if district_column in input_data.columns:
        input_data[district_column] = 1


    # Occupation
    occupation_column = "Occupation_" + occupation

    if occupation_column in input_data.columns:
        input_data[occupation_column] = 1


    # -------------------------
    # MAKE PREDICTION
    # -------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    yes_probability = probabilities[1]


    # -------------------------
    # DISPLAY RESULT
    # -------------------------

    st.divider()

    st.subheader("🎯 Prediction Result")

    if prediction == 1:

        st.success(
            "### YES — The pupil is predicted to have listened "
            "to SRHR messages."
        )

    else:

        st.error(
            "### NO — The pupil is predicted not to have listened "
            "to SRHR messages."
        )


    st.metric(
        "Probability of listening",
        f"{yes_probability * 100:.1f}%"
    )

    st.caption(
        "Prediction generated using the trained Random Forest model."
    )
