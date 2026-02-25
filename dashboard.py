import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Set page configuration
st.set_page_config(page_title="Cancer Risk Analysis Dashboard", layout="wide", page_icon="🏥")

# Theme / Title
st.title("🏥 Cancer Risk Analysis Dashboard")

# Load Dataset
@st.cache_data
def load_data():
    file_path = "cancer patient data sets - Sheet.csv"
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Drop Patient Id as it's not a feature
            if 'Patient Id' in df.columns:
                df = df.drop(columns=['Patient Id'])
            return df
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Cache the model training
@st.cache_resource
def train_model(df):
    try:
        # Encode Level for training
        le = LabelEncoder()
        df_train = df.copy()
        df_train['Level'] = le.fit_transform(df_train['Level'])

        X = df_train.drop(columns=['Level'])
        y = df_train['Level']

        # Train a basic model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, le, X.columns.tolist()
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

with st.spinner("Loading data..."):
    df = load_data()

if df is not None:
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a Section", ["Quick Overview", "Exploratory Data Analysis", "Risk Level Prediction"])

    if app_mode == "Quick Overview":
        st.header("📊 Dataset Overview")
        st.write("First 10 rows of the dataset:")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Summary Statistics:**")
            st.write(df.describe())
        with col2:
            st.write("**Missing Values Check:**")
            null_counts = df.isnull().sum()
            if null_counts.sum() == 0:
                st.success("No missing values found!")
            else:
                st.write(null_counts[null_counts > 0])

    elif app_mode == "Exploratory Data Analysis":
        st.header("📈 Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Cancer Risk Levels")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.countplot(x='Level', data=df, palette='viridis', order=['Low', 'Medium', 'High'], ax=ax1)
            st.pyplot(fig1)

        with col2:
            st.subheader("Age Distribution of Patients")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(df['Age'], bins=20, kde=True, color='skyblue', ax=ax2)
            st.pyplot(fig2)

        st.subheader("Correlation Heatmap of Risk Factors")
        # Preprocessing for heatmap: Encode Level if it's categorical
        df_encoded = df.copy()
        temp_le = LabelEncoder()
        if 'Level' in df_encoded.columns:
            df_encoded['Level'] = temp_le.fit_transform(df_encoded['Level'])
        
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)

        st.subheader("Interactive Feature Comparison")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            feature_x = st.selectbox("Select X-axis Feature", numeric_cols, index=numeric_cols.index('Age') if 'Age' in numeric_cols else 0)
        with col_c2:
            feature_y = st.selectbox("Select Y-axis Feature", numeric_cols, index=numeric_cols.index('Air Pollution') if 'Air Pollution' in numeric_cols else 1)
        with col_c3:
            plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Box Plot", "Violin Plot", "Bar Plot"])
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Scatter Plot":
            sns.scatterplot(x=feature_x, y=feature_y, hue='Level', data=df, palette='Set1', ax=ax4)
        elif plot_type == "Box Plot":
            sns.boxplot(x='Level', y=feature_y, data=df, palette='viridis', order=['Low', 'Medium', 'High'], ax=ax4)
            st.info(f"Showing distribution of **{feature_y}** across different Risk Levels.")
        elif plot_type == "Violin Plot":
            sns.violinplot(x='Level', y=feature_y, data=df, palette='magma', order=['Low', 'Medium', 'High'], ax=ax4)
            st.info(f"Showing distribution and density of **{feature_y}** across different Risk Levels.")
        elif plot_type == "Bar Plot":
            sns.barplot(x='Level', y=feature_y, data=df, palette='coolwarm', order=['Low', 'Medium', 'High'], ax=ax4)
            st.info(f"Showing average **{feature_y}** for each Risk Level.")
            
        st.pyplot(fig4)

    elif app_mode == "Risk Level Prediction":
        st.header("🔮 Cancer Risk Level Prediction")
        st.write("Input patient details to predict the risk level.")

        model, le, feature_names = train_model(df)

        if model:
            # UI for User Inputs
            st.subheader("Patient Details")
            user_inputs = {}
            
            # Divide inputs into columns
            cols = st.columns(3)
            for i, column in enumerate(feature_names):
                with cols[i % 3]:
                    if column == 'Gender':
                        user_inputs[column] = st.selectbox(f"{column} (1:Male, 2:Female)", options=[1, 2])
                    elif column == 'Age':
                        user_inputs[column] = st.number_input(f"{column}", min_value=1, max_value=120, value=30)
                    else:
                        min_val = float(df[column].min())
                        max_val = float(df[column].max())
                        mean_val = float(df[column].mean())
                        user_inputs[column] = st.slider(f"{column}", min_val, max_val, mean_val)

            if st.button("Predict Risk Level"):
                input_df = pd.DataFrame([user_inputs])
                prediction = model.predict(input_df)
                prediction_label = le.inverse_transform(prediction)[0]
                
                st.markdown("---")
                if prediction_label == 'High':
                    st.error(f"Predicted Risk Level: **{prediction_label}**")
                    st.subheader("💡 Recommended Actions & Advice")
                    st.write("""
                    - **Urgent Medical Consultation**: Please schedule an appointment with an oncologist or your primary care physician immediately for professional screening.
                    - **Diagnostic Testing**: Consider discussing Low-Dose CT scans or other relevant diagnostic tests with your doctor.
                    - **Symptom Monitoring**: Keep a detailed log of symptoms like persistent cough, chest pain, or unexplained weight loss.
                    - **Strict Lifestyle Changes**: If you smoke, seek immediate professional help to quit. Minimize exposure to air pollution and secondhand smoke.
                    """)
                elif prediction_label == 'Medium':
                    st.warning(f"Predicted Risk Level: **{prediction_label}**")
                    st.subheader("💡 Recommended Actions & Advice")
                    st.write("""
                    - **Routine Screening**: Ensure you are up-to-date with your annual physical exams and discuss your risk factors with a doctor.
                    - **Lifestyle Modification**: Focus on reducing alcohol consumption and increasing physical activity.
                    - **Environmental awareness**: Try to improve indoor air quality and wear masks in highly polluted areas.
                    - **Dietary Improvement**: Incorporate more antioxidant-rich foods like fruits and leafy greens into your diet.
                    """)
                else:
                    st.success(f"Predicted Risk Level: **{prediction_label}**")
                    st.subheader("💡 Recommended Actions & Advice")
                    st.write("""
                    - **Health Maintenance**: Continue your healthy habits! Regular exercise and a balanced diet are key.
                    - **Regular Check-ups**: Maintain standard annual wellness visits to monitor overall health.
                    - **Prevention**: Stay informed about the long-term effects of environmental factors and maintain a smoke-free lifestyle.
                    - **Awareness**: Even with low risk, stay mindful of any unusual or persistent changes in your health.
                    """)
                
                st.info("⚠️ **Note**: This is a machine learning demonstration based on historical patterns and is **NOT** a substitute for professional medical advice, diagnosis, or treatment.")
else:
    st.error("Dataset not loaded. Please check the file path and format.")
