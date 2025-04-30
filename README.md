# Transfer-Learning-Experiment-in-CNNs
This repository contains the implementation and experiments for Assignment 1 in image classification using convolutional neural networks (CNNs) and transfer learning. The goal is to evaluate the effectiveness of transfer learning by comparing training strategies using the Cats vs Dogs dataset and the Stanford Dogs dataset.
Overview
We conducted four experiments using the same CNN architecture and hyperparameters, modifying different parts of the network to test the impact of transfer learning.

🔬 Overview

Experiment 1:
Train a CNN model from scratch on the Cats vs Dogs dataset using a learning rate of 0.0001.

Experiment 2:
Train a model on the Stanford Dogs dataset, save it, and then replace only the output layer before fine-tuning on the Cats vs Dogs dataset.

Experiment 3:
Replace the output layer and the first two convolutional layers from the pretrained Stanford Dogs model before training on the Cats vs Dogs dataset.

Experiment 4:
Replace the output layer and the last two convolutional layers, keeping all other weights from the Stanford Dogs model, and train on the Cats vs Dogs dataset.

All models were trained for 50 epochs, and accuracy was recorded for each epoch.

📁 Project Structure

├── experiments/
│   ├── experiment_1_train_from_scratch.ipynb
│   ├── experiment_2_finetune_output_layer.ipynb
│   ├── experiment_3_finetune_first_layers.ipynb
│   ├── experiment_4_finetune_last_layers.ipynb
│
├── datasets/
│   ├── cats_vs_dogs/        # Custom downloaded cats and dogs dataset
│   ├── stanford_dogs/       # Downloaded using TensorFlow Datasets
│
├── models/
│   ├── stanford_model.h5    # Saved model trained on Stanford Dogs
│
├── results/
│   ├── experiment_*.csv     # Accuracy logs for each experiment
│
├── report.pdf               # Short report with results and discussion
├── README.md


⚙️ Requirements
Python 3.8+
TensorFlow 2.x
NumPy
Matplotlib
scikit-learn (optional for further metrics)

🧪 Results Summary

Experiment	Best Accuracy	Notes
1	XX%	Training from scratch
2	XX%	Transfer learning: output layer replaced
3	XX%	Transfer learning: output + first 2 conv layers replaced
4	XX%	Transfer learning: output + last 2 conv layers replaced
📄 See report.pdf for detailed plots and analysis.
