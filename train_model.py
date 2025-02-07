import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ✅ Step 1: Load Your Dataset (Assuming 'teaminfo.csv' contains training data)
df = pd.read_csv("teaminfo.csv")

# ✅ Step 2: Select Features & Target Variable
df = df[['year', 'athletes', 'age', 'prev_medals', 'medals']]  # Keep only relevant columns

# Remove missing values
df.dropna(inplace=True)

# ✅ Step 3: Define Input (X) and Target (y)
X = df[['year', 'athletes', 'age', 'prev_medals']]  # Features
y = df['medals']  # Target (medal count)

# ✅ Step 4: Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Step 6: Save the Model as a .pkl File
joblib.dump(model, "olympic_medal_predictor.pkl")

print("✅ Model trained and saved as 'olympic_medal_predictor.pkl'!")
