# /// script
# requires-python = ">=3.9"
# dependencies = [
#  "pandas",
#  "lightgbm",
#  "scikit-learn"
# ]
# description = "A script to use the h5 model to generate inference"
# entry-point = "ripeness_determinant.py"
# ///

"""
This script loads a pre-trained model from a pickle file and uses it to generate predictions based on input data.

The script defines a function `predict_label()` that accepts input data, ensures it matches the expected format, and uses the loaded model to predict the label.

Usage:
1. Load the pre-trained model from a pickle file.
2. Prepare input data as a list of dictionaries where each dictionary represents a sample with keys: 
   - 'N' (Nitrogen content)
   - 'P' (Phosphorus content)
   - 'K' (Potassium content)
   - 'temperature' (Temperature in Celsius)
   - 'humidity' (Humidity percentage)
   - 'ph' (Soil pH level)
   - 'rainfall' (Rainfall in millimeters)
3. Pass the input data to the `predict_label()` function for prediction.
4. The function returns the predicted label based on the model's inference.

Example usage:
    input_data = [{
        'N': 30, 
        'P': 40, 
        'K': 50, 
        'temperature': 25, 
        'humidity': 75, 
        'ph': 6.5, 
        'rainfall': 100
    }]
    predicted_label = predict_label(input_data)
    print(f"Predicted label: {predicted_label}")
    
Note:
- The model expects the input data to have specific columns: 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'.
- The output is a prediction based on the input data, which can be used for further analysis or decision-making.
"""


import pickle
import pandas as pd

# Load the trained model from the pickle file
model_path = 'NPK_model.pkl'  
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Function to predict the label based on input features
def predict_label(input_data):
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame(input_data)

    # Ensure the input has the same columns as the model expects
    expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_df = input_df[expected_columns]

    # Make the prediction
    prediction = model.predict(input_df)

    return prediction

# Example usage
if __name__ == "__main__":
    # Example input data
    input_data = [{
        'N': 30, 
        'P': 40, 
        'K': 50, 
        'temperature': 25, 
        'humidity': 75, 
        'ph': 6.5, 
        'rainfall': 100
    }]

    # Predict the label
    predicted_label = predict_label(input_data)

    print(f"Predicted label: {predicted_label}")

