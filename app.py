from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib  # To load the trained model
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("olympic_medal_predictor.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # Loads the UI page

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the user
    input_data = request.json
    year = float(input_data['year'])
    athletes = float(input_data['athletes'])
    age = float(input_data['age'])
    prev_medals = float(input_data['prev_medals'])

    # Convert into NumPy array
    features = np.array([[year, athletes, age, prev_medals]])

    # Predict using the trained model
    prediction = model.predict(features)

    # Return the predicted medals
    return jsonify({"predicted_medals": round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
