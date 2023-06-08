import tensorflow as tf
from tensorflow import keras
from keras import layers

# Load the flower dataset
data_dir = "flower_photos/flower_photos/"
image_size = (180, 180)
batch_size = 32

# Create a data generator for loading and augmenting the images
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
)

# Load and split the dataset into training and validation sets
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create the CNN model
model = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(180, 180, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(val_data, verbose=2)
print("Test accuracy:", test_acc)
