# AI_Emergency_Intelligence_System\app\app.py
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="SentinelAI", layout="wide")

# ------------------------------------------------
# UI STYLE
# ------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.glow-title {
    font-size:50px;
    color:#38bdf8;
    text-align:center;
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 10px #38bdf8; }
    to { text-shadow: 0 0 25px #0ea5e9; }
}
.stMetric {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='glow-title'>🛡 SentinelAI</div>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Smart Urban Emergency Intelligence Platform</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Predict. Prevent. Protect.</h4>", unsafe_allow_html=True)

# ------------------------------------------------
# PATHS
# ------------------------------------------------
BASE_PATH = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "models", "emergency_risk_model.pkl")
DATA_PATH = os.path.join(BASE_PATH, "data_processed", "final_emergency_hourly_dataset.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=100)
st.sidebar.title("SentinelAI Control Panel")
section = st.sidebar.radio(
    "Navigate",
    ["About SentinelAI", "Live Intelligence", "Upload Dataset"]
)

# ==========================================================
# ABOUT SECTION
# ==========================================================
if section == "About SentinelAI":

    st.markdown("## 🚀 Vision")
    st.write("""
    SentinelAI is an AI-powered emergency intelligence system
    designed to analyze urban crime, crash, and weather data
    to predict high-risk hours in cities.
    """)

    st.markdown("## 🎯 Purpose")
    st.write("""
    - Optimize emergency deployment
    - Improve public safety planning
    - Enable predictive risk monitoring
    """)

    st.markdown("## ⚙ AI Workflow")
    st.write("""
    Data Cleaning → Feature Engineering → Random Forest Model →
    Risk Classification → Explainable AI Output
    """)

# ==========================================================
# LIVE INTELLIGENCE
# ==========================================================
elif section == "Live Intelligence":

    st.markdown("## 📊 Real-Time Emergency Risk Analysis")

    col1, col2 = st.columns(2)

    selected_date = col1.selectbox("Select Date", df["date"].unique())
    selected_hour = col2.selectbox("Select Hour", sorted(df["hour"].unique()))

    sample = df[(df["date"] == selected_date) & (df["hour"] == selected_hour)]

    if not sample.empty:

        features = ["crime_count", "crash_count", "PRCP", "TAVG", "is_peak_hour"]

        prediction = model.predict(sample[features])[0]
        probabilities = model.predict_proba(sample[features])[0]
        confidence = max(probabilities) * 100

        # Risk Banner
        if prediction == "High":
            st.error("🚨 HIGH RISK HOUR DETECTED")
        elif prediction == "Medium":
            st.warning("⚠ MODERATE RISK HOUR")
        else:
            st.success("✅ LOW RISK HOUR")

        st.markdown(f"### 🤖 AI Confidence Score: **{confidence:.2f}%**")

        # Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Crime Count", int(sample["crime_count"]))
        m2.metric("Crash Count", int(sample["crash_count"]))
        m3.metric("Rainfall", float(sample["PRCP"]))
        m4.metric("Temperature", float(sample["TAVG"]))
        m5.metric("Peak Hour", int(sample["is_peak_hour"]))

        st.markdown("---")

        # ----------------------------------------------------
        # Dynamic Charts
        # ----------------------------------------------------
        st.markdown("## 📊 Intelligent Risk Visualization")

        day_data = df[df["date"] == selected_date]

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("### 📈 Risk Distribution (Selected Day)")
            fig1, ax1 = plt.subplots()
            ax1.hist(day_data["emergency_risk_score"], bins=15)
            selected_score = sample["emergency_risk_score"].values[0]
            ax1.axvline(selected_score, linestyle='dashed')
            ax1.set_xlabel("Emergency Risk Score")
            ax1.set_ylabel("Frequency")
            st.pyplot(fig1)
            fig1.savefig("risk_distribution.png")

        with col_chart2:
            st.markdown("### ⚖ Crime vs Crash (Selected Day)")
            fig2, ax2 = plt.subplots()
            ax2.scatter(day_data["crime_count"], day_data["crash_count"])
            ax2.scatter(
                sample["crime_count"].values[0],
                sample["crash_count"].values[0],
                s=200
            )
            ax2.set_xlabel("Crime Count")
            ax2.set_ylabel("Crash Count")
            st.pyplot(fig2)
            fig2.savefig("crime_vs_crash.png")

        st.markdown("---")

        # Top Influencing Factor
        importances = model.feature_importances_
        feature_importance_dict = dict(zip(features, importances))
        top_feature = max(feature_importance_dict, key=feature_importance_dict.get)

        # AI Reasoning Box
        st.markdown("## 🧠 AI Reasoning")

        st.markdown(f"""
        <div style='background-color:#1e293b;padding:20px;border-radius:15px;color:white;'>
        SentinelAI analyzed crime density, traffic intensity,
        weather volatility, and peak-hour dynamics.

        The dominant influencing factor for this hour is
        <strong>{top_feature.replace('_',' ').title()}</strong>.

        Based on multi-factor risk synthesis,
        this hour is classified as <strong>{prediction}</strong>
        with a confidence level of {confidence:.2f}%.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ----------------------------------------------------
        # PDF REPORT
        # ----------------------------------------------------
        if st.button("📄 Generate Downloadable AI Report"):

            report_path = "SentinelAI_Report.pdf"
            doc = SimpleDocTemplate(report_path)
            elements = []
            styles = getSampleStyleSheet()

            elements.append(Paragraph("SentinelAI Emergency Intelligence Report", styles["Title"]))
            elements.append(Spacer(1, 0.5 * inch))

            elements.append(Paragraph(f"Date: {selected_date}", styles["Normal"]))
            elements.append(Paragraph(f"Hour: {selected_hour}", styles["Normal"]))
            elements.append(Paragraph(f"Predicted Risk Level: {prediction}", styles["Normal"]))
            elements.append(Paragraph(f"Confidence: {confidence:.2f}%", styles["Normal"]))

            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Image("risk_distribution.png", width=5*inch, height=3*inch))
            elements.append(Spacer(1, 0.3 * inch))
            elements.append(Image("crime_vs_crash.png", width=5*inch, height=3*inch))

            doc.build(elements)

            with open(report_path, "rb") as f:
                st.download_button("⬇ Download Report", f, file_name="SentinelAI_Report.pdf")

# ==========================================================
# UPLOAD DATASET
# ==========================================================
elif section == "Upload Dataset":

    st.markdown("## 📂 Upload Your Dataset")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(new_df.head())