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

