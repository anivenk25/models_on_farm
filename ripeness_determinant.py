# /// script
# requires-python = ">=3.7, <3.12"
# dependencies = [
#   "numpy<2",
#   "tensorflow-cpu>=2.10.0,<2.13.0",  # Adjust TensorFlow version based on your requirements
#   "pillow>=8.0.0",
#   "statistics"
# ]
# description = "A script to use the h5 model to generate inference"
# entry-point = "ripeness_determinant.py"
# ///

"""
This script loads a pre-trained TensorFlow model and uses it to predict the ripeness of fruits based on their RGB images.

The script processes an image by:
1. Loading and resizing the image.
2. Downsampling the image by selecting every N-th pixel.
3. Preprocessing each pixel to match the model's input format.
4. Making predictions for each pixel.
5. Aggregating the predictions using the mode of the classes.
6. Computing the confidence based on the prediction probabilities.

Classes:
- "Early Ripe"
- "Partially Ripe"
- "Ripe"
- "Decay"

Main steps:
1. Load the pre-trained model from the file `fruit_ripeness_rgb_model.h5`.
2. Prompt the user for the path to an image and validate the input.
3. Resize and convert the image to RGB format.
4. Downsample the image for faster processing.
5. Process each pixel of the downsampled image, predict its class, and store the predictions.
6. After processing 100 pixels, aggregate the predictions to get the most frequent class (mode).
7. Compute the confidence by finding the maximum prediction probability for the selected pixels.
8. Output the final prediction along with the confidence level.

Usage:
- Run the script, input the path to the image when prompted, and receive the ripeness prediction and confidence.

Example:
    python ripeness_determinant.py
    Enter the path to the image: /path/to/image.jpg
    Final Prediction: Ripe (Confidence: 0.85)

Note:
- The script processes only a downsampled subset of pixels to speed up the prediction process. You can adjust the downsampling factor or the pixel count as needed.
- Ensure the image is in a valid format and accessible at the provided path.
"""

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import sys
import statistics as stats

# Load the model
MODEL_PATH = 'fruit_ripeness_rgb_model.h5'
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    sys.exit(1)

# Classes for prediction output
CLASSES = ["Early Ripe", "Partially Ripe", "Ripe", "Decay"]

# Preprocessing function for a single pixel
def preprocess_pixel(pixel):
    """
    Preprocess a single pixel to fit the model's input requirements.
    """
    try:
        pixel = np.array(pixel)
        pixel = np.expand_dims(pixel, axis=0)
        return pixel
    except Exception as e:
        print(f"Error preprocessing pixel: {e}")
        sys.exit(1)

# Main script
def main():
    # Prompt user to input image path
    image_path = input("Enter the path to the image: ").strip()

    if not os.path.isfile(image_path):
        print("Error: File not found!")
        return

    # Open image and convert it to RGB
    try:
        img = Image.open(image_path).convert('RGB')  # Ensure image is RGB
        img = img.resize((128,128))  # Resize to a much smaller size for faster processing
        img = np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)

    # Downsample the image by skipping pixels (every N-th pixel)
    downsampled_img = img[::10, ::10]  # Skip every 10th pixel (downsample by 10)

    # Prepare for predictions
    predictions = []

    # Process image pixel by pixel
    for i in range(downsampled_img.shape[0]):
        for j in range(downsampled_img.shape[1]):
            pixel = downsampled_img[i, j]
            preprocessed_pixel = preprocess_pixel(pixel)

            # Make prediction for this pixel
            try:
                prediction = model.predict(preprocessed_pixel)
                predicted_class = np.argmax(prediction)
                print(predicted_class)
                predictions.append(predicted_class)
                
                # Early exit after processing 100 pixels (for speed)
                if len(predictions) >= 100:
                    break
            except Exception as e:
                print(f"Error during prediction for pixel ({i}, {j}): {e}")
                continue

        if len(predictions) >= 100:
            break  # Exit after processing 100 pixels

   # Aggregate the predictions (use mode for most frequent label)
    final_prediction = int(stats.mode(predictions))  # Correct usage of mode

    # Get the predicted class name from CLASSES
    predicted_class_name = CLASSES[final_prediction]

    # Compute the confidence (average confidence for the selected pixels)
    confidence = np.max([np.max(model.predict(preprocess_pixel(downsampled_img[i, j]))) for i in range(downsampled_img.shape[0]) for j in range(downsampled_img.shape[1])])

    print(f"Final Prediction: {predicted_class_name} (Confidence: {confidence:.2f})")

if __name__ == '__main__':
    main()

