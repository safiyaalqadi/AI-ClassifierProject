import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'flower_classifier_model.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
else:
    print(f"Error: Model file not found at {model_path}")
    exit(1)

def process_image(image):
    # Convert the image to a TensorFlow tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Resize the image to 224x224
    image = tf.image.resize(image, (224, 224))
    
    # Normalize the pixel values to the range [0, 1]
    image /= 255.0
    
    # Convert back to a NumPy array
    return image.numpy()

def predict(image_path, model, top_k=5):
    # Load and preprocess the image
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Make predictions
    predictions = model.predict(processed_image)
    
    # Get the top K predictions
    top_k_probs, top_k_indices = tf.math.top_k(predictions, k=top_k)
    
    # Convert indices to class labels
    top_k_probs = top_k_probs.numpy().flatten()
    top_k_indices = top_k_indices.numpy().flatten()
    top_k_classes = [str(index) for index in top_k_indices]
    
    return top_k_probs, top_k_classes

def load_labels(label_path):
    with open(label_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def plot_predictions(image_path, model, class_names, top_k=5):
    # Predict the top K classes
    probs, classes = predict(image_path, model, top_k)
    
    # Load the image
    im = Image.open(image_path)
    
    # Create a subplot with the input image and the bar chart of probabilities
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)
    
    # Plot the input image
    ax1.imshow(im)
    ax1.axis('off')
    ax1.set_title('Input Image')
    
    # Plot the probabilities as a bar chart
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([class_names[str(cls)] for cls in classes])
    ax2.invert_yaxis()  # Invert y-axis to have the highest probability on top
    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Predictions')
    
    plt.tight_layout()
    plt.show()

def main(image_path, top_k, category_names):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        exit(1)
    
    # Load class names if category names file is provided
    if category_names:
        class_names = load_labels(category_names)
    else:
        class_names = None
    
    # Make predictions
    probs, classes = predict(image_path, model, top_k)

    # Output the predictions
    print("Predictions for the image:")
    for i in range(top_k):
        class_name = class_names[str(classes[i])] if class_names else classes[i]
        print(f"{i+1}: {class_name} with probability {probs[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Flower Class from an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to a JSON file mapping labels to flower names')
    args = parser.parse_args()

    main(args.image_path, args.top_k, args.category_names)
