import tensorflow
from tensorflow.keras.datasets import mnist
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the shapes of the datasets
print(f"x_train is: {x_train.shape}")
print(f"y_train is: {y_train.shape}")
print(f"x_test is: {x_test.shape}")
print(f"y_test is: {y_test.shape}")

# Plot sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

# Pre-compute the counts
unique, counts = np.unique(y_train, return_counts=True)
counts_dict = dict(zip(unique, counts))

print("Digital Counts: ", counts_dict)

# Plot the distribution of classes using Seaborn
counts_df = pd.DataFrame(list(counts_dict.items()), columns=['Digits', 'Frequency'])
plt.figure(figsize=(10, 5))
sns.barplot(x='Digits', y='Frequency', data=counts_df)
plt.title("Distribution of classes in training data")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.show()

# Plot the distribution of classes using Matplotlib (only use if you don't have Seaborn)
plt.figure(figsize=(10, 5))
plt.bar(unique, counts, tick_label=unique)
plt.title("Distribution of Classes in Training Data")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.show()

# Check for missing values
print(f"Missing values in x_train: {np.isnan(x_train).sum()}")
print(f"Missing values in y_train: {np.isnan(y_train).sum()}")
print(f"Missing values in x_test: {np.isnan(x_test).sum()}")
print(f"Missing values in y_test: {np.isnan(y_test).sum()}")

# Flatten the images to analyze pixel values
x_train_flat = x_train.reshape(-1)

plt.figure(figsize=(10, 5))
plt.hist(x_train_flat, bins=255, color='blue', alpha=0.7)
plt.title("Pixel Value Distribution in training Data")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# Normalize pixel values to the range [0,1]
x_train_normalized = x_train/255.0
x_test_normalized = x_test/255.0

# Reshape the images to 28*28*1 (adding a channel dimension)
x_train_reshaped = x_train_normalized.reshape(-1, 28, 28, 1)
x_test_reshaped = x_test_normalized.reshape(-1, 28, 28, 1)


# def create_model():  # optimizer='adam', dropout_rate=0.0
# Initialize the model
model = Sequential()

# Add first convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add first max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add second convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Add second max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps
model.add(Flatten())

# Add first fully connected (Dense) layer
model.add(Dense(128, activation='relu'))

# Add dropout layer
model.add(Dropout(0.5))

# Add output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# param_grid = {
#     'batch_size': [32],
#     'epochs': [10],
#     'optimizer': ['adam', 'rmsprop'],
# }

# Hyperparameter tuning
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(x_train_reshaped, y_train)
#
# print(f'Best : {grid_result.best_score_} using {grid_result.best_params_}')
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, std, param in zip(means, stds, params):
#     print(f"{mean} (+/-{std}) with : {param}")

# Define the data augmentation parameters
# datagen = ImageDataGenerator(
#     rotation_range=10,            # Random rotations between 0 and 10 degrees
#     width_shift_range=0.1,     # Random Horizontal shifts
#     height_shift_range=0.1,   # Random Vertical shifts
#     zoom_range=0.1,            # Random zoom
#     horizontal_flip=True,       # Random Horizontal flips
#     fill_mode='nearest'        # Fill in Missing pixels after transformation
# )

# Fit the data augmentation on the training data
# datagen.fit(x_train_reshaped)

# # Reinitialize and train the model with augmented data
# best_model = create_model(optimizer=best_params['optimizer'], dropout_rate=best_params['dropout_rate'])


# Split the training data into training and validation sets
x_train_final, x_val, y_train_final, y_val = train_test_split(x_train_reshaped, y_train, test_size=0.2, random_state=42)

# Use early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the Model with augmented data
history = model.fit(x_train_final, y_train_final,
                         epochs=10,
                         batch_size=32,
                         validation_data=(x_val, y_val),)

                         # steps_per_epoch=len(x_train_final) // 32,
                         # callbacks=[early_stopping])

# Print the model summary
model.summary()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict the labels for the test set
y_pred = model.predict(x_test_reshaped)
y_pred_classes = y_pred.argmax(axis=1)

# Generate a classification report
report = classification_report(y_test, y_pred_classes, target_names=[str(i) for i in range(10)])

# Plot training and validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion')
plt.show()

# Visualize some test predictions
num_images = 10
random_indices = np.random.choice(x_test_reshaped.shape[0], num_images, replace=False)
x_display = x_test_reshaped[random_indices]
y_display_true = y_test[random_indices]
y_display_pred = y_pred_classes[random_indices]

plt.figure(figsize=(15, 5))
for i in range(num_images):
    plt.subplot(2, num_images // 2, i + 1)
    plt.imshow(x_display[i].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_display_true[i]}, Pred: {y_display_pred[i]}')
    plt.axis("off")
plt.show()

# Save the trained model to a file for later use
model.save("digit_recognition_model.keras")
