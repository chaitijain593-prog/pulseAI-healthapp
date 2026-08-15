import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# ---------------------------------------------------------
# Page Configuration & Subtle Dark Medical Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="PulseAI | Clinical Health Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Fix Padding, Smooth Curved Sidebar Selection & Dark Medical Theme
st.markdown(
    """
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* 1. REMOVE TOP WHITE SPACE & STREAMLIT DEFAULT PADDING */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 1;
    }
    
    .stAppViewContainer > .main {
        padding-top: 1rem !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* Global Dark Background */
    .stApp {
        background: #090d16;
        background-image: radial-gradient(at 0% 0%, rgba(13, 148, 136, 0.12) 0px, transparent 50%),
                          radial-gradient(at 100% 100%, rgba(14, 165, 233, 0.08) 0px, transparent 50%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #e2e8f0;
    }

    /* 2. PAGE HEADER BANNER */
    .health-header {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        padding: 1.6rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
    }

    .health-header h1 {
        color: #f9fafb !important;
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .health-header p {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* 3. CURVED SIDEBAR SELECTION FIX */
    .nav-link-selected {
        border-top-left-radius: 10px !important;
        border-bottom-left-radius: 10px !important;
        border-top-right-radius: 10px !important;
        border-bottom-right-radius: 10px !important;
        overflow: hidden !important;
    }

    /* 4. EXPANDER DARK MODE STYLING */
    div[data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
        margin-bottom: 1.5rem !important;
    }

    div[data-testid="stExpander"] summary {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stExpander"] summary:hover {
        color: #2dd4bf !important;
    }

    /* 5. INPUT FIELD OVERRIDES */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
        border-radius: 10px !important;
        border: 1px solid #374151 !important;
    }

    .stNumberInput label, .stSelectbox label {
        color: #9ca3af !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* 6. RESULT CARDS */
    .metric-card-danger {
        background: rgba(225, 29, 72, 0.12);
        border: 1px solid rgba(225, 29, 72, 0.4);
        border-left: 5px solid #f43f5e;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        color: #fecdd3;
        margin-bottom: 1.5rem;
    }

    .metric-card-success {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 5px solid #10b981;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        color: #a7f3d0;
        margin-bottom: 1.5rem;
    }

    /* 7. PRIMARY ACTION BUTTON */
    div.stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.7rem 1.8rem;
        border-radius: 10px;
        border: 1px solid rgba(45, 212, 191, 0.3);
        width: 100%;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.25);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        box-shadow: 0 6px 18px rgba(20, 184, 166, 0.35);
    }

    /* 8. SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #0d131f !important;
        border-right: 1px solid #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

working_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# Load Saved Models & Scalers
# ---------------------------------------------------------
diabetes_model = None
custom_threshold = 0.40

try:
    diabetes_model = joblib.load(
        os.path.join(working_dir, "diabetes_svm_pipeline.pkl")
    )
    try:
        custom_threshold = joblib.load(
            os.path.join(working_dir, "diabetes_svm_threshold.pkl")
        )
    except Exception:
        custom_threshold = 0.40
except Exception:
    pass

heart_model = None
heart_scaler = None

try:
    heart_model = joblib.load(os.path.join(working_dir, "heart_model.pkl"))
    heart_scaler = joblib.load(os.path.join(working_dir, "scaler.pkl"))
except Exception:
    pass


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 0.5rem 0;">
            <h2 style="color: #38bdf8; margin: 0; font-size: 1.3rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                🩺 PulseAI <span style="font-size: 0.7rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 2px 8px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3);">PRO</span>
            </h2>
            <p style="color: #6b7280; font-size: 0.82rem; margin-top: 2px;">Clinical Diagnostics Engine</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["Diabetes Prediction", "Heart Disease Prediction"],
        icons=["activity", "heart-pulse"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#14b8a6", "font-size": "16px"},
            "nav-link": {
                "font-size": "13.5px",
                "text-align": "left",
                "margin": "4px 0px",
                "color": "#9ca3af",
                "border-radius": "10px",
                "padding": "9px 12px",
            },
            "nav-link-selected": {
                "background-color": "#1f2937",
                "color": "#38bdf8",
                "font-weight": "600",
                "border-left": "4px solid #14b8a6",
                "border-top-left-radius": "10px",
                "border-bottom-left-radius": "10px",
                "border-top-right-radius": "10px",
                "border-bottom-right-radius": "10px",
            },
        },
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="background: rgba(31, 41, 55, 0.6); padding: 12px; border-radius: 10px; border: 1px solid #374151;">
            <p style="color: #9ca3af; font-size: 0.8rem; margin: 0; line-height: 1.4;">
                🛡️ <b>Clinical Assistant</b><br>Trained Machine Learning diagnostic screening tool.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Page 1: Diabetes Prediction
# ---------------------------------------------------------
if selected == "Diabetes Prediction":
    st.markdown(
        """
        <div class="health-header">
            <h1>🩸 Diabetes Assessment</h1>
            <p>Evaluate metabolic parameters using Support Vector Machine classification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Clinical Reference Guidelines", expanded=False):
        st.markdown(
            """
            * **Fasting Glucose:** Normal < 100 mg/dL | Prediabetes: 100–125 mg/dL | Diabetes >= 126 mg/dL
            * **BMI Index:** Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese >= 30
            * **Blood Pressure:** Normal Diastolic < 80 mmHg
            """
        )

    st.markdown("<h4 style='color:#f3f4f6; font-size:1.05rem; margin-bottom:0.8rem;'>Patient Laboratory & Physical Indicators</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Pregnancies Count", min_value=0, max_value=20, value=1)
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.38, step=0.01)

    with col2:
        Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        Insulin = st.number_input("Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=79)
        Age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

    with col3:
        BloodPressure = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=180, value=70)
        BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("RUN DIABETES EVALUATION"):
        if diabetes_model is None:
            st.error("Model file missing! Please ensure 'diabetes_svm_pipeline.pkl' is in the project folder.")
        else:
            try:
                g = float(Glucose) if float(Glucose) != 0 else np.nan
                bp = float(BloodPressure) if float(BloodPressure) != 0 else np.nan
                st_val = float(SkinThickness) if float(SkinThickness) != 0 else np.nan
                ins = float(Insulin) if float(Insulin) != 0 else np.nan
                bmi_val = float(BMI) if float(BMI) != 0 else np.nan

                user_input = [
                    float(Pregnancies),
                    g,
                    bp,
                    st_val,
                    ins,
                    bmi_val,
                    float(DiabetesPedigreeFunction),
                    float(Age),
                ]

                prob = diabetes_model.predict_proba([user_input])[0][1]
                risk_percentage = round(prob * 100, 1)
                is_diabetic = prob >= custom_threshold

                st.markdown("<h3 style='color:#f9fafb; font-size:1.2rem; margin-top:0.5rem;'>Diagnostic Assessment Result</h3>", unsafe_allow_html=True)

                if is_diabetic:
                    st.markdown(
                        f"""
                        <div class="metric-card-danger">
                            <h4 style="margin:0; font-size:1.15rem; font-weight:700;">⚠️ Elevated Risk: Diabetic Indicators Present</h4>
                            <p style="margin:4px 0 0 0; font-size:0.95rem;">Estimated Probability: <b>{risk_percentage}%</b> (Threshold: {int(custom_threshold*100)}%)</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="metric-card-success">
                            <h4 style="margin:0; font-size:1.15rem; font-weight:700;">✅ Low Risk: Non-Diabetic Profile</h4>
                            <p style="margin:4px 0 0 0; font-size:0.95rem;">Estimated Probability: <b>{risk_percentage}%</b> (Threshold: {int(custom_threshold*100)}%)</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                v_col1, v_col2 = st.columns([1, 1])

                with v_col1:
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=risk_percentage,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Diabetes Risk Score (%)", "font": {"size": 15, "color": "#9ca3af"}},
                            number={"font": {"color": "#f9fafb", "size": 36}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                                "bar": {"color": "#f43f5e" if is_diabetic else "#14b8a6"},
                                "bgcolor": "#1f2937",
                                "bordercolor": "#374151",
                                "steps": [
                                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.15)"},
                                    {"range": [65, 100], "color": "rgba(244, 63, 94, 0.15)"},
                                ],
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=290,
                        margin=dict(l=20, r=20, t=30, b=20),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with v_col2:
                    radar_categories = ["Glucose", "BMI", "Blood Pressure", "Insulin", "Age"]
                    raw_vals = [Glucose/200*100, BMI/50*100, BloodPressure/140*100, Insulin/300*100, Age/80*100]

                    fig_radar = go.Figure(
                        data=[
                            go.Scatterpolar(
                                r=raw_vals,
                                theta=radar_categories,
                                fill="toself",
                                fillcolor="rgba(20, 184, 166, 0.2)",
                                line=dict(color="#14b8a6", width=2),
                                name="Patient Profile",
                            )
                        ],
                        layout=go.Layout(
                            title=dict(text="Metabolic Parameter Overview", font=dict(size=14, color="#9ca3af")),
                            polar=dict(
                                bgcolor="#1f2937",
                                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#6b7280")),
                                angularaxis=dict(tickfont=dict(color="#e5e7eb", size=10)),
                            ),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                            height=290,
                            margin=dict(l=40, r=40, t=30, b=20),
                        ),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")


# ---------------------------------------------------------
# Page 2: Heart Disease Prediction
# ---------------------------------------------------------
if selected == "Heart Disease Prediction":
    st.markdown(
        """
        <div class="health-header">
            <h1>🫀 Cardiovascular Health Assessment</h1>
            <p>Analyze key clinical cardiac indicators using machine learning models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Clinical Field Descriptions & References", expanded=False):
        st.markdown(
            """
            * **trestbps:** Resting blood pressure on admission (mm Hg).
            * **chol:** Serum cholesterol level (mg/dl) [Desirable: < 200 mg/dl].
            * **thalach:** Maximum heart rate achieved during testing.
            * **oldpeak:** ST depression induced by exercise relative to rest.
            * **ca:** Major vessels (0-4) colored by fluoroscopy.
            """
        )

    st.markdown("<h4 style='color:#f3f4f6; font-size:1.05rem; margin-bottom:0.8rem;'>Patient Cardiac Measurements</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=52)
        sex = st.selectbox("Biological Sex", options=["Male (1)", "Female (0)"])
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[
                "0: Typical Angina",
                "1: Atypical Angina",
                "2: Non-anginal Pain",
                "3: Asymptomatic",
            ],
        )
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=240, value=125)
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=212)

    with col2:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=["0: False", "1: True"],
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[
                "0: Normal",
                "1: ST-T Wave Abnormality",
                "2: Left Ventricular Hypertrophy",
            ],
        )
        thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=230, value=168)
        exang = st.selectbox(
            "Exercise Induced Angina",
            options=["0: No", "1: Yes"],
        )

    with col3:
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST",
            options=["0: Upsloping", "1: Flat", "2: Downsloping"],
        )
        ca = st.selectbox(
            "Major Vessels Colored (ca)",
            options=["0", "1", "2", "3", "4"],
        )
        thal = st.selectbox(
            "Thalassemia Status",
            options=[
                "0: Normal",
                "1: Fixed Defect",
                "2: Reversible Defect",
                "3: Unknown",
            ],
            index=2,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("RUN CARDIAC EVALUATION"):
        if heart_model is None or heart_scaler is None:
            st.error("Model or Scaler missing! Verify 'heart_model.pkl' and 'scaler.pkl' are present in the folder.")
        else:
            try:
                sex_val = 1 if "Male" in sex else 0
                cp_val = int(cp.split(":")[0])
                fbs_val = int(fbs.split(":")[0])
                restecg_val = int(restecg.split(":")[0])
                exang_val = int(exang.split(":")[0])
                slope_val = int(slope.split(":")[0])
                ca_val = int(ca)
                thal_val = int(thal.split(":")[0])

                user_data = [
                    float(age),
                    sex_val,
                    cp_val,
                    float(trestbps),
                    float(chol),
                    fbs_val,
                    restecg_val,
                    float(thalach),
                    exang_val,
                    float(oldpeak),
                    slope_val,
                    ca_val,
                    thal_val,
                ]

                scaled_data = heart_scaler.transform([user_data])
                probs = heart_model.predict_proba(scaled_data)[0]
                risk_percentage = round(probs[1] * 100, 1) if len(probs) > 1 else (100.0 if heart_model.predict(scaled_data)[0] == 1 else 0.0)

                st.markdown("<h3 style='color:#f9fafb; font-size:1.2rem; margin-top:0.5rem;'>Diagnostic Assessment Result</h3>", unsafe_allow_html=True)

                if risk_percentage >= 50.0:
                    st.markdown(
                        f"""
                        <div class="metric-card-danger">
                            <h4 style="margin:0; font-size:1.15rem; font-weight:700;">⚠️ High Risk: Cardiac Indications Detected</h4>
                            <p style="margin:4px 0 0 0; font-size:0.95rem;">Estimated Probability: <b>{risk_percentage}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="metric-card-success">
                            <h4 style="margin:0; font-size:1.15rem; font-weight:700;">✅ Low Risk: Normal Cardiac Profile</h4>
                            <p style="margin:4px 0 0 0; font-size:0.95rem;">Estimated Probability: <b>{risk_percentage}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                v_col1, v_col2 = st.columns([1, 1])

                with v_col1:
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=risk_percentage,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Heart Risk Score (%)", "font": {"size": 15, "color": "#9ca3af"}},
                            number={"font": {"color": "#f9fafb", "size": 36}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                                "bar": {"color": "#f43f5e" if risk_percentage >= 50.0 else "#14b8a6"},
                                "bgcolor": "#1f2937",
                                "bordercolor": "#374151",
                                "steps": [
                                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.15)"},
                                    {"range": [65, 100], "color": "rgba(244, 63, 94, 0.15)"},
                                ],
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=290,
                        margin=dict(l=20, r=20, t=30, b=20),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with v_col2:
                    cardiac_df = pd.DataFrame({
                        'Marker': ['BP (mmHg)', 'Cholesterol', 'Max HR'],
                        'Patient Value': [trestbps, chol, thalach],
                        'Normal Target': [120, 200, 150]
                    })

                    fig_bar = go.Figure(data=[
                        go.Bar(name='Patient Value', x=cardiac_df['Marker'], y=cardiac_df['Patient Value'], marker_color='#14b8a6'),
                        go.Bar(name='Normal Benchmark', x=cardiac_df['Marker'], y=cardiac_df['Normal Target'], marker_color='#374151')
                    ])
                    fig_bar.update_layout(
                        barmode='group',
                        title=dict(text="Biomarkers vs Standard Benchmarks", font=dict(size=14, color="#9ca3af")),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(font=dict(color="#e5e7eb")),
                        xaxis=dict(tickfont=dict(color="#e5e7eb")),
                        yaxis=dict(tickfont=dict(color="#9ca3af"), gridcolor="rgba(255,255,255,0.05)"),
                        height=290,
                        margin=dict(l=20, r=20, t=30, b=20),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")