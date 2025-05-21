# Transfer-Learning-Experiment-in-CNNs

The objective of the assignment is to explore transfer learning in Convolutional Neural Networks (CNNs) through a series of experiments focused on layer replacement. 

The network architecture and training parameters are kept consistent across the experiments to ensure valid comparisons. 

The changes in each experiment and their impact on the results will be documented in this comprehensive report

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

![image](https://github.com/user-attachments/assets/86a51e06-1eb5-4218-bfd7-3f3d869e860a)





⚙️ Requirements

Python 3.8+

TensorFlow 2.x

NumPy

scikit-learn (optional for further metrics)

🧪 Results Summary

Experiment	Best Validation Accuracy	Notes

1)	44%	Training from scratch

2)	73%	Transfer learning: output layer replaced

3）	79%	Transfer learning: output + first 2 conv layers replaced

4）	87%	Transfer learning: output + last 2 conv layers replaced

📄 See report.pdf for detailed plots and analysis.
