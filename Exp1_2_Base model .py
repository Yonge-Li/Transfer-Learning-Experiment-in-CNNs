import os
import numpy as np
import keras
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow import data as tf_data
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image_dataset_from_directory
import argparse
from tensorflow.data import AUTOTUNE
import argparse
import csv

# Hyperparameters
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 50

# DATA_PATH = "/home/liyo23ac/assignment1/PetImages"
# Argument parsing
parser = argparse.ArgumentParser(description='Train a CNN model on the Stanford Dogs dataset')
parser.add_argument('log_dir', type=str, help='Directory to save logs')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
args = parser.parse_args()


def build_stanford_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    #x = layers.Rescaling(1.0 / 255)(inputs)
    x = inputs
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def main():
    # 1. loading the Stanford Dogs dataset
    train_ds,info = tfds.load(
        'stanford_dogs',
        split='train',
        with_info=True,
        as_supervised=True
    )
    test_ds = tfds.load(
        'stanford_dogs', 
        split='test', 
        as_supervised=True
    )

    #num_classes = info.features['label'].num_classes
    # 2. Preprocessing function
    
    def preprocess_train(image, label):
    
        larger_size = (int(IMAGE_SIZE[0] * 1.2), int(IMAGE_SIZE[1] * 1.2))
        image = tf.image.resize(image, larger_size)  
    
        image = tf.image.random_crop(image, size=(*IMAGE_SIZE, 3))
    
        image = tf.image.random_flip_left_right(image)

        image = tf.cast(image, tf.float32) / 255.0
        return image, label


    def preprocess(image, label):
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(preprocess_train).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
   
    # 3. modeling
    num_classes = info.features['label'].num_classes
    model = build_stanford_model(IMAGE_SIZE + (3,), num_classes)
    
    # 4. training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
       

    # Define callbacks for saving models
    callbacks = [        
        keras.callbacks.ModelCheckpoint(  
            "best_stanford_dogs_model.keras",  
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
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=callbacks
    )      

    txt_path = "stanford dogs model_results.txt"
    csv_path = "stanford dogs model_results.csv"


    with open(txt_path, "w") as txt_file, open(csv_path, "w", newline='') as csv_file:
    
        writer = csv.writer(csv_file)
    
        writer.writerow(["Epoch", "Train Acc", "Val Acc", "Train Loss", "Val Loss"])

    
        print("\nstanford dogs model Training Results:")
        txt_file.write("stanford dogs model Training Results:\n")

        for epoch in range(len(history.history['accuracy'])):
            train_acc = history.history['accuracy'][epoch]
            val_acc = history.history['val_accuracy'][epoch]
            train_loss = history.history['loss'][epoch]
            val_loss = history.history['val_loss'][epoch]

            line = (f"Epoch {epoch+1:2d} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}")
        
        
            print(line)
       
            txt_file.write(line + "\n")
        
            writer.writerow([epoch+1, train_acc, val_acc, train_loss, val_loss])

      


if __name__ == "__main__":
    main()