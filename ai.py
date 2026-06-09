# Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Display Dataset Shape
print("Training Data Shape:", x_train.shape)
print("Testing Data Shape :", x_test.shape)

# Display Few Images from Dataset
plt.figure(figsize=(10,5))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build Simple Neural Network
model = Sequential([
    Flatten(input_shape=(28,28)),   # Convert 2D image to 1D
    Dense(128, activation='relu'),  # Hidden Layer
    Dense(10, activation='softmax') # Mlti-Class Classification
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
).

# Evaluate Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_accuracy)

# Predict on Test Data
predictions = model.predict(x_test)

# Display Prediction Results
plt.figure(figsize=(10,5))

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[i], cmap='gray')

    predicted_label = np.argmax(predictions[i])

    plt.title(f"Pred: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Print Actual vs Predicted
for i in range(5):
    predicted_label = np.argmax(predictions[i])

    print(f"Actual: {y_test[i]}  |  Predicted: {predicted_label}")