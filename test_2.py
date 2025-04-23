import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("Plant_disease_model_3.keras")

# Define the class labels mapping for 3 classes.
# Ensure that the order corresponds to how the model was trained.
class_labels = {
    0: "Healthy",
    1: "Diseased: Bacterial Disease",
    2: "Diseased: Manganese Toxicity"
}

def predict_image(img_path):
    # Load the image with the correct target size (224x224)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    # Add batch dimension and normalize the image
    img_array = np.expand_dims(img_array, axis=0) / 255.0  
    prediction = model.predict(img_array)
    # Use np.argmax to select the index with highest probability
    pred_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_labels[pred_index]
    return predicted_class

def plot_images_with_predictions(image_paths):
    n = len(image_paths)
    cols = 3  # Number of columns in the grid
    rows = (n + cols - 1) // cols  # Compute number of rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, img_path in enumerate(image_paths):
        # Load and normalize the image (for display)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        pred = predict_image(img_path)
        axes[i].imshow(img_array)
        axes[i].set_title(f"{pred}", fontsize=16, color='Crimson')
        axes[i].axis("off")

    # Hide any extra subplots if n < rows*cols
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Add a main title for the entire figure
    fig.suptitle("Plant Disease Predictions", fontsize=20, y=0.98)
    plt.show()


# List of image paths to test
image_paths = [
    'Output_images/B1.jpg',
    'Output_images/G19.jpeg',
    'Output_images/B8.jpg',
    'Output_images/G11.jpg',
    'Output_images/B15.jpg',
    'Output_images/G5.jpg',
    'Output_images/B10.jpg',
    'Data Set/Healthy/Healthy_58.jpg',
    'Output_images/B9.jpg',   
]

plot_images_with_predictions(image_paths)
