import os
import numpy as np
import keras
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow import data as tf_data
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_datasets as tfds
import argparse


# Argument parsing
parser = argparse.ArgumentParser(description='Train a CNN model on the Stanford Dogs dataset')
parser.add_argument('log_dir', type=str, help='Directory to save logs')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
args = parser.parse_args()


# Update the dataset path to point to the server directory
dataset_path = "/home/liyo23ac/assignment1/PetImages" 
# Function to check and remove corrupted images
def remove_corrupted_images(dataset_path):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(dataset_path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                # Try to open the image to check if it's corrupt
                img = tf.io.read_file(fpath)
                img = tf.image.decode_jpeg(img, channels=3)
            except tf.errors.InvalidArgumentError:  # In case of corrupt JPEG
                num_skipped += 1
                os.remove(fpath)  # Delete the corrupt file
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                
    print(f"Deleted {num_skipped} corrupted images.")
# Run this function before loading the dataset
remove_corrupted_images(dataset_path)

image_size = (180, 180)
batch_size = 64

train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training", 
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)



# Data augmentation layers
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Apply data augmentation
inputs = keras.Input(shape=(image_size + (3,)))
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 255)(x)

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

print("Loading stanford_dogs_model.")
# Load pre-trained model
pretrained = keras.models.load_model("Best_Stanford_Dogs_Model.keras", compile=False)

# Replace output layer
x = pretrained.layers[-2].output
new_output = layers.Dense(1, activation="sigmoid")(x)
model_exp4 = keras.Model(pretrained.input, new_output)

# Identify convolutional layers
conv_layers = [layer for layer in model_exp4.layers 
               if isinstance(layer, (layers.Conv2D, layers.SeparableConv2D))]

# Unfreeze last two convolutional layers
for layer in conv_layers[-2:]:
    layer.trainable = True
for layer in model_exp4.layers:
    if layer not in conv_layers[-2:] and layer != model_exp4.layers[-1]:
        layer.trainable = False


print("Compiling and training:")

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# Define callbacks for saving models
callbacks = [        
    keras.callbacks.ModelCheckpoint(  
        os.path.join(args.log_dir,"best_epx3_model.keras"),  
        save_best_only=True,  
        monitor="val_accuracy"
        ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        mode="max"
    )]  

# Compile and train
model_exp4.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)
history_exp4 = model_exp4.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=callbacks
)

import csv


txt_path = "Experiment4_results.txt"
csv_path = "Experiment4_results.csv"


with open(txt_path, "w") as txt_file, open(csv_path, "w", newline='') as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow(["Epoch", "Train Acc", "Val Acc", "Train Loss", "Val Loss"])


    print("\nExperiment 4: replace the output layer and the two last convolutional layers:")
    txt_file.write("Experiment 4: replace the output layer and the two last convolutional layers:\n")

    for epoch in range(len(history_exp4.history['accuracy'])):
        train_acc = history_exp4.history['accuracy'][epoch]
        val_acc = history_exp4.history['val_accuracy'][epoch]
        train_loss = history_exp4.history['loss'][epoch]
        val_loss = history_exp4.history['val_loss'][epoch]

        line = (f"Epoch {epoch+1:2d} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}")
    
    
        print(line)

        txt_file.write(line + "\n")
    
        writer.writerow([epoch+1, train_acc, val_acc, train_loss, val_loss])