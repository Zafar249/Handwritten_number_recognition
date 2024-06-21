import tensorflow
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("C:\\Users\\Zafar\\Documents\\Stuff\\python codes\\"
                   "projects\\Handwritten Number Recognition\\digit_recognition_model.keras")


# Function to preprocess the image
def preprocess_image(image_path):
    # Open the image using PIL
    img = Image.open(image_path).convert("L")  # Convert Image to grayscale
    img = img.resize((28, 28))  # Resize image to 28 * 28 pixels
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0  # Normalize the image
    return img_array


def predict_digit(image_path):  # Predict the digit in the given image
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit


# Image path
image_path = ("C:\\Users\\Zafar\\Documents\\Stuff\\python codes\\projects\\"
              "Handwritten Number Recognition\\sample images\\four.jpeg")
# Predict the digit
predicted_digit = predict_digit(image_path)
print(f'Predicted_digit: {predicted_digit}')
