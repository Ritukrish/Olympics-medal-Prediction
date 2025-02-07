# Olympics-medal-Prediction
1.	Olympic Medal Prediction Project

Objective:
The goal of this project was to predict the number of Olympic medals a country might win in upcoming Olympic events based on historical data of past medal counts, athlete participation, and other influencing factors.

Dataset:
The dataset consisted of historical Olympic records with features such as:
Country name,Year of participation,Number of athletes,Previous medal counts,Athletes', age and experience, Past Olympic performances

Data Preprocessing:
•	Handling Missing Values: Missing values were filled using the mean, median, and mode techniques.
•	Removing Outliers: Identified and removed inconsistent records affecting model accuracy.
•	Feature Selection: Retained only relevant features such as country, year, previous medals, and number of athletes.
•	Feature Scaling: Applied Min-Max Scaling and Standardization to normalize data for better model performance.
•	Encoding Categorical Data: Converted country names into numerical values using one-hot encoding.

Model Used:
Linear Regression Model:
•	Used to predict the medal count based on the selected features.
•	Trained using 80% training data and 20% test data.

Performance Evaluation:
•	Root Mean Square Error (RMSE): Used to measure the difference between actual and predicted medal counts.
•	R² Score: Evaluated how well the model fits the data.
•	Heatmaps & Correlation Analysis: Used to understand relationships between features.

Results & Findings:
Countries with higher athlete participation had a higher probability of winning medals.
Past performance played a significant role in future medal wins.
Linear regression successfully predicted medal counts with a reasonable accuracy.
The model could be improved further by incorporating more data points, external economic factors, and training techniques.
