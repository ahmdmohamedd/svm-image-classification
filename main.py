# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10

# Step 1: Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Step 2: Preprocess the data
# Flatten the images (32x32x3) into 1D arrays for SVM input
x_train_flatten = x_train.reshape(x_train.shape[0], -1)
x_test_flatten = x_test.reshape(x_test.shape[0], -1)

# Normalize the data (scale pixel values to range [0, 1])
x_train_flatten = x_train_flatten / 255.0
x_test_flatten = x_test_flatten / 255.0

# Flatten the labels from 2D to 1D
y_train = y_train.flatten()
y_test = y_test.flatten()

# Step 3: Use a smaller subset for faster training
subset_size = 0.1  # Use 10% of the data for quick training

x_train_sub, _, y_train_sub, _ = train_test_split(
    x_train_flatten, y_train, train_size=subset_size, stratify=y_train, random_state=42)
x_test_sub, _, y_test_sub, _ = train_test_split(
    x_test_flatten, y_test, train_size=subset_size, stratify=y_test, random_state=42)

# Step 4: Train the SVM model
# Create an SVM model with an RBF kernel
svm_model = SVC(kernel='rbf', gamma='auto')

# Train the SVM model on the subset of data
print("Training the SVM model...")
svm_model.fit(x_train_sub, y_train_sub)

# Step 5: Evaluate the model
# Predict on the test subset
y_pred = svm_model.predict(x_test_sub)

# Print classification report
print("Classification Report:\n", classification_report(y_test_sub, y_pred))

# Print confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test_sub, y_pred))

# Step 6: Visualize some test images with predicted and true labels
def plot_images(images, true_labels, pred_labels, n_images=10):
    plt.figure(figsize=(10, 5))
    for i in range(n_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].reshape(32, 32, 3))
        plt.title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
        plt.axis('off')
    plt.show()

# Plot the first 10 test images, their true labels, and predicted labels
plot_images(x_test_sub[:10], y_test_sub[:10], y_pred[:10])
