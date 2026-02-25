# 🏥 Cancer Risk Analysis & Prediction Dashboard

An interactive web application built with **Streamlit** to analyze cancer patient data and predict cancer risk levels using **Machine Learning**. This project provides behavioral, environmental, and clinical insights to help understand factors contributing to lung cancer risk.

## 🚀 Live Demo
*(Once deployed, add your Streamlit Cloud link here)*

## ✨ Features

- **📊 Comprehensive Data Exploration (EDA)**:
  - **Distribution Analysis**: Visualize risk levels across different demographics.
  - **Age Demographics**: Histogram with KDE to understand the age range of patients.
  - **Correlation Heatmap**: Analyze how factors like Smoking, Obesity, and Air Pollution correlate with one another and the risk level.

- **📈 Interactive Visualizations**:
  - Choice of **Scatter Plots, Box Plots, Violin Plots, and Bar Charts**.
  - Customizable X and Y axes to compare specific health factors.

- **🔮 Risk Level Prediction**:
  - Uses a **Random Forest Classifier** trained on the patient dataset.
  - Real-time prediction of **Low, Medium, or High** risk based on user inputs.
  - **Personalized Advice**: Specific health and lifestyle recommendations based on the predicted risk level.

- **⚡ Optimized Experience**:
  - **Caching**: Data loading and model training are cached for instant performance.
  - **Responsive Design**: Clean and professional UI suitable for researchers and students.

## 🛠️ Tech Stack

- **Frontend/Dashboard**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
- **Visualizations**: [Seaborn](https://seaborn.pydata.org/) & [Matplotlib](https://matplotlib.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Random Forest)

## 📂 Project Structure

```bash
├── dashboard.py                       # Main Streamlit application
├── cancer patient data sets - Sheet.csv # The dataset used for analysis/training
├── requirements.txt                   # Dependencies for deployment
└── README.md                          # Project documentation
```

## ⚙️ Installation & Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/cancer-risk-analysis.git
   cd cancer-risk-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run dashboard.py
   ```

## 🧪 Dataset Information

The dataset includes 1000 records of patients with 25 clinical and behavioral features, including:
- **Demographics**: Age, Gender.
- **Environmental**: Air Pollution, Occupational Hazards, Dust Allergy.
- **Lifestyle**: Alcohol use, Smoking, Balanced Diet, Obesity.
- **Symptoms**: Chest Pain, Coughing of Blood, Fatigue, Weight Loss, Shortness of Breath.

## ⚠️ Disclaimer

This dashboard is for **educational and demonstration purposes only**. The predictions are based on historical data patterns and **not** on real-time clinical diagnosis. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with a healthcare professional for clinical concerns.

---
*Created with ❤️ by [Your Name/Handle]*