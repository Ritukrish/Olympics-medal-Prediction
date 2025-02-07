import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load trained model
model = joblib.load("olympic_medal_predictor.pkl")

# Streamlit UI
st.title("🏅 Olympic Medal Prediction")

# User inputs
year = st.number_input("Year:", min_value=1900, max_value=2030, step=1)
athletes = st.number_input("Number of Athletes:", min_value=0, step=1)
age = st.number_input("Average Age:", min_value=10, max_value=50, step=1)
prev_medals = st.number_input("Previous Medals:", min_value=0, step=1)

if st.button("Predict Medals"):
    features = np.array([[year, athletes, age, prev_medals]])
    prediction = model.predict(features)
    st.success(f"🏆 Predicted Medals: {round(prediction[0], 2)}")

# Load dataset
df = pd.read_csv("teaminfo.csv")

# Data Visualization
st.subheader("📊 Data Correlation Heatmap")
corr_matrix = df[['year', 'athletes', 'age', 'prev_medals', 'medals']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
st.pyplot(plt)

st.subheader("📉 Pairplot Visualization")
st.pyplot(sns.pairplot(data=df))
