# 🛡 SentinelAI
### Smart Urban Emergency Intelligence Platform  
**Predict • Prevent • Protect**

SentinelAI is an AI-powered emergency intelligence system designed to analyze urban data such as crime incidents, traffic crashes, and weather conditions to predict high-risk emergency hours in cities.

The system helps improve public safety by identifying potential emergency hotspots and providing intelligent insights for decision-makers.

---

# 🚀 Project Overview

Modern cities generate massive amounts of data related to crime, traffic incidents, and environmental conditions. SentinelAI integrates these datasets and applies machine learning to detect patterns and predict emergency risk levels.

The platform provides:

• AI-based emergency risk prediction  
• Intelligent data visualization  
• Real-time risk monitoring  
• Automated AI-generated reports  

---

# 🎯 Objectives

- Predict high-risk emergency hours in urban areas
- Assist authorities in proactive emergency response
- Improve city safety through data-driven intelligence
- Demonstrate real-world AI application in public safety systems

---

# 🧠 AI Model

SentinelAI uses a **Random Forest Classifier** to predict risk levels based on multiple urban indicators.

### Input Features

- Crime Count
- Traffic Crash Count
- Rainfall (PRCP)
- Temperature (TAVG)
- Peak Hour Indicator

### Output

The AI predicts three risk categories:

- Low Risk
- Medium Risk
- High Risk

The system also calculates a **confidence score** for each prediction.

---

# 📊 Dataset Sources

The project uses real-world public datasets:

• Chicago Crime Data – Chicago Open Data Portal  
• Traffic Crash Data – Chicago Transportation Data  
• Weather Data – NOAA National Centers for Environmental Information  
• Fire Station Data – Chicago Government Open Data

---

# ⚙ Project Architecture

```
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Machine Learning Model
   ↓
Risk Prediction
   ↓
Interactive Dashboard
```

---

# 🖥 System Features

### AI Risk Prediction
Predict emergency risk levels for any selected date and hour.

### Intelligent Visualization
Dynamic charts showing:

• Risk score distribution  
• Crime vs crash correlation  

### AI Reasoning Engine
Displays the top influencing factor behind predictions.

### Automated AI Reports
Generate downloadable **PDF intelligence reports**.

### Dataset Upload
Users can upload their own datasets for analysis.

---

# 📂 Project Structure

```
SentinelAI

app/
    app.py

models/
    emergency_risk_model.pkl

src/
    data_cleaning.py
    feature_engineering.py
    risk_model.py
    utils.py

requirements.txt
README.md
```

---

# 💻 Installation

Clone the repository

```
git clone https://github.com/Radhika-12s/SentinelAI-Emergency-Intelligence.git
```

Move into the project directory

```
cd SentinelAI
```

Install required libraries

```
pip install -r requirements.txt
```

Run the application

```
streamlit run app/app.py
```

---

# 📈 Example Output

SentinelAI dashboard provides:

• AI risk prediction  
• Confidence score  
• Emergency indicators  
• Dynamic visualizations  
• Downloadable intelligence reports  

---

# 🌍 Potential Real-World Applications

• Smart City Infrastructure  
• Emergency Response Optimization  
• Police Deployment Planning  
• Traffic Safety Monitoring  
• Disaster Preparedness  

---

# 🧑‍💻 Technologies Used

Python  
Streamlit  
Pandas  
Scikit-learn  
Matplotlib  
ReportLab  

---

# ⭐ Future Improvements

• Real-time API integration  
• Deep learning prediction models  
• Geospatial hotspot mapping  
• Live city dashboard deployment  

---

# 👨‍💻 Author

Developed as an **AI-driven urban safety intelligence system** as part of a computer science project.
