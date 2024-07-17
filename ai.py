# TODO: Make all necessary imports.
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import json
import scipy

# TODO: Load the dataset with TensorFlow Datasets.
dataset, info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
# TODO: Create a training set, a validation set and a test set.
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

# TODO: Get the number of examples in each set from the dataset info.
num_train_examples = info.splits['train'].num_examples
num_val_examples = info.splits['validation'].num_examples
num_test_examples = info.splits['test'].num_examples

# TODO: Get the number of classes in the dataset from the dataset info
num_classes = info.features['label'].num_classes

print(f'Training examples: {num_train_examples}')
print(f'Validation examples: {num_val_examples}')
print(f'Test examples: {num_test_examples}')
print(f'Number of classes: {num_classes}')



# TODO: Print the shape and corresponding label of 3 images in the training set.
for image, label in train_dataset.take(3):
    print(f'Image shape: {image.shape}, Label: {label.numpy()}')
with open('label_map.json', 'r') as f:
    class_names = json.load(f)


    # TODO: Plot 1 image from the training set. Set the title 
# of the plot to the corresponding image label. 
for image, label in train_dataset.take(1):
    plt.imshow(image)
    plt.title(f'Label: {label.numpy()}')
    plt.axis('off')
    plt.show()

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

    # TODO: Plot 1 image from the training set. Set the title 
# of the plot to the corresponding class name. 
for image, label in train_dataset.take(1):
    plt.imshow(image)
    plt.title(f'Class Name: {class_names[str(label.numpy())]}')
    plt.axis('off')
    plt.show()

# TODO: Create a pipeline for each set.
def format_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_dataset.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_dataset.map(format_image).batch(BATCH_SIZE).prefetch(1)


# TODO: Build and train your network.
mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(
    mobilenet_url,
    input_shape=(224, 224, 3),
    trainable=False
)

# Define the new feed-forward classifier
model = tf.keras.Sequential([
    tf.keras.layers.Lambda(feature_extractor, input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(102, activation='softmax') 
])
# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define the number of epochs
EPOCHS = 10

# Train the model
history = model.fit(
    train_batches,
    epochs=EPOCHS,
    validation_data=validation_batches
)



# TODO: Plot the loss and accuracy values achieved during training for the training and validation set.
# Plot the loss and accuracy values achieved during training for the training and validation set
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



# TODO: Print the loss and accuracy values achieved on the entire test set.
test_loss, test_accuracy = model.evaluate(test_batches)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')


# TODO: Save your trained model as a Keras model.
model.save('flower_classifier_model.keras')



# TODO: Load the Keras model
loaded_model = tf.keras.models.load_model('flower_classifier_model.keras', custom_objects={'KerasLayer': hub.KerasLayer})

# Verify the model structure
loaded_model.summary()


# TODO: Create the process_image function
from PIL import Image
import numpy as np
import tensorflow as tf

def process_image(image):
    # Convert the image to a TensorFlow tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Resize the image to 224x224
    image = tf.image.resize(image, (224, 224))
    
    # Normalize the pixel values to the range [0, 1]
    image /= 255.0
    
    # Convert back to a NumPy array
    return image.numpy()


from PIL import Image

image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()



# TODO: Create the predict function
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

# Example usage
image_path = './test_images/hard-leaved_pocket_orchid.jpg'
probs, classes = predict(image_path, loaded_model, top_k=5)
print(probs)
print(classes)



# TODO: Plot the input image along with the top 5 classes
import matplotlib.pyplot as plt

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

# Example usage
image_path = './test_images/hard-leaved_pocket_orchid.jpg'
plot_predictions(image_path, loaded_model, class_names, top_k=5)


