# Transfer-Learning-Experiment-in-CNNs
The objective of the assignment is to explore transfer learning in Convolutional Neural Networks (CNNs) through a series of experiments focused on layer replacement. The network architecture and training parameters are kept consistent across the experiments to ensure valid comparisons. The changes in each experiment and their impact on the results will be documented in this comprehensive report

Overview
We started based on a good tutorial on how to build an image classifier from scratch: https://keras.io/examples/vision/image_classification_from_scratch

then we conducted four experiments using the same CNN architecture and hyperparameters, modifying different parts of the network to test the impact of transfer learning.

🔬 Overview

Experiment 1-1:
Train a CNN model from scratch on the Cats vs Dogs dataset using a learning rate of 0.0001.

Experiment 1-2:
Train a model on the Stanford Dogs dataset, save the best one.

Experiment 2:
Replace only the output layer before fine-tuning on the Cats vs Dogs dataset.

Experiment 3:
Replace the output layer and the first two convolutional layers from the pretrained Stanford Dogs model before training on the Cats vs Dogs dataset.

Experiment 4:
Replace the output layer and the last two convolutional layers, keeping all other weights from the Stanford Dogs model, and train on the Cats vs Dogs dataset.

All models were trained for 50 epochs, and accuracy was recorded for each epoch.

📁 Project Structure

├── experiments/
│   ├── Exp1-1_tutoring_model.py
│   ├── Exp1-2_base_model.py
│   ├── Exp2_finetune_output_layer.py
│   ├── Exp3_finetune_first_layers.py
│   ├── Exp4_finetune_last_layers.py
│
├── datasets/
│   ├── cats_vs_dogs/        # Custom downloaded cats and dogs dataset
│   ├── stanford_dogs/       # Downloaded using TensorFlow Datasets
│
├── models/
│   ├── stanford_model.h5    # Saved model trained on Stanford Dogs
│
├── results/
│   ├── Exp1_1_CatsDogs_results.txt                 
│   ├── Exp1_2_stanford_dogs_model_results.txt
│   ├── Experiment2_results.txt
│   ├── Experiment3_results.txt
│   ├── Experiment4_results.txt
│
├── report.pdf               
├── README.md


⚙️ Requirements
Python 3.8+
TensorFlow 2.x
NumPy
scikit-learn (optional for further metrics)

🧪 Results Summary

Experiment	Best Accuracy	Notes
1	XX%	Training from scratch
2	XX%	Transfer learning: output layer replaced
3	XX%	Transfer learning: output + first 2 conv layers replaced
4	XX%	Transfer learning: output + last 2 conv layers replaced
📄 See report.pdf for detailed plots and analysis.
