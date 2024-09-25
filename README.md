# SVM Image Classification on CIFAR-10 Dataset

This project implements **Support Vector Machines (SVM)** for image classification using the **CIFAR-10 dataset**. SVM is a supervised machine learning algorithm that is used for classification or regression tasks. In this project, we use the **CIFAR-10** dataset to classify images into 10 different categories.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to apply SVM to classify images from the CIFAR-10 dataset. CIFAR-10 consists of 60,000 32x32 color images in 10 different categories, such as airplanes, cars, birds, cats, and more. Since SVM tends to work better on small-scale datasets, we use a smaller subset of CIFAR-10 for faster training and evaluation. The implementation leverages Python libraries such as `scikit-learn` and `keras`.

## Dataset

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The 10 classes include:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

For this project, we use the **Keras** library to load the dataset.

## Installation

Follow the steps below to run this project on your local machine:

1. Clone the repository:
    ```bash
    git clone https://github.com/ahmdmohamedd/svm-image-classification.git
    cd svm-image-classification
    ```

2. Ensure that you have Python 3.x installed on your system. You can check the version with:
    ```bash
    python --version
    ```

3. Required Python packages:
    - `numpy`
    - `matplotlib`
    - `scikit-learn`
    - `keras`
    - `tensorflow` (if you prefer to use TensorFlow as the backend for Keras)

## Usage

Once the dependencies are installed, you can run the script to train and evaluate the SVM model on the CIFAR-10 dataset.

1. **Training the Model:**

    Simply run the `main.py` script:
    ```bash
    python main.py
    ```

2. **Visualizing the Results:**

    The script will output a classification report, confusion matrix, and a visualization of the first 10 test images along with their true and predicted labels.

## Results

After training the SVM model on a subset of CIFAR-10, the model will output the following:
- A **classification report** showing precision, recall, f1-score, and support for each class.
- A **confusion matrix** showing the number of correct and incorrect predictions for each class.
- A plot showing some test images with their **true labels** and **predicted labels**.

Example classification report:
```
              precision    recall  f1-score   support

           0       0.80      0.75      0.77       100
           1       0.90      0.85      0.88       100
           ...
```

## Project Structure

```bash
.
├── main.py   # Main script for SVM training and testing
├── README.md                     # Project documentation
└── LICENSE                       # License file (optional)
```

## Contributing

Contributions are welcome! If you have any suggestions or find a bug, feel free to open an issue or submit a pull request.
