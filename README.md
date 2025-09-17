# Face Mask Detection with CNN

## 🎭 Project Overview

This project implements a face mask detection system using a **Convolutional Neural Network (CNN)**. The model is trained on a custom dataset to classify faces into three categories: **`with_mask`**, **`without_mask`**, and **`mask_weared_incorrect`**. The notebook details the entire process, from data parsing and preprocessing to model training, evaluation, and deployment via a simple API interface using **Gradio**.

## 🚀 Key Features

  * **Data Parsing**: The code includes functions to parse XML annotations and load the data into a pandas DataFrame.
  * **Image Preprocessing**: Faces are cropped from the original images based on bounding box data and then resized and normalized for the model.
  * **Data Augmentation**: The training data is augmented using techniques like horizontal flips, zooming, and rotations to improve model robustness.
  * **CNN Architecture**: A sequential CNN model is built with multiple convolutional and pooling layers, followed by dropout and dense layers.
  * **Model Training**: The model is trained using the **Adam optimizer** and monitored with **Early Stopping** and **ReduceLROnPlateau** callbacks to prevent overfitting and optimize the learning rate.
  * **Model Evaluation**: The model's performance is evaluated using metrics like **accuracy, recall, precision, and AUC**.
  * **Interactive Interface**: A user-friendly web interface is created with **Gradio**, allowing users to upload new images and receive real-time predictions.
  * **Deployment Testing**: A Flask model is also provided for testing deployment. This allows you to verify the model's functionality in a production-like environment.

## 📁 Repository Structure

  * `Face Mask_Detection_Notebook.ipynb`: The main Jupyter Notebook containing all the code.
  * `Datasets/Face Detection/`: The directory where the image and XML annotation files are expected to be located.
  * `my_model.h5` and `my_model.keras`: The saved trained model files.
  * **`Flask Directory`**: This directory contains the necessary files to run the Flask application for deployment testing. You can find the link to this directory in the notebook's documentation.

## ⚙️ Prerequisites

To run this notebook, you need to set up a Python environment with the following libraries. You can install them using `pip`:

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn Pillow gradio
```

## 📊 Data and Preprocessing

The dataset consists of images and corresponding XML annotation files that specify the bounding boxes and labels for each face.

1.  **Parsing Annotations**: A function reads the XML files to extract the image file names, bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`), and labels (`name`).
2.  **Label Distribution**: The labels are highly imbalanced, with `with_mask` being the most common.
3.  **Cropping and Saving**: The code crops each face from the images based on the bounding boxes and saves them into separate directories for `train`, `validation`, and `test` sets, organized by their respective labels.
4.  **Data Augmentation**: An `ImageDataGenerator` is used on the training data to apply random transformations like flipping, zooming, and rotation, which helps the model generalize better.

## 🧠 CNN Architecture and Training

The model uses a simple yet effective CNN architecture:

  * **Input Layer**: A `Conv2D` layer with 16 filters, a `3x3` kernel, and `ReLU` activation, expecting a `35x35x3` input image.
  * **Hidden Layers**: Two more `Conv2D` layers with `MaxPooling2D` are used to extract features. A `Dropout` layer is added to reduce overfitting.
  * **Output Layer**: A `Flatten` layer is followed by two `Dense` layers with `ReLU` activation and a final `Dense` layer with `softmax` activation for multiclass classification. `l2` regularization is also applied to one of the dense layers.
  * **Training**: The model is compiled with the `adam` optimizer and `categorical_crossentropy` loss. Callbacks for `EarlyStopping` and `ReduceLROnPlateau` are used to optimize the training process.

## 📈 Evaluation Results

After training, the model's performance is evaluated on the test set.

  * **Loss**: \~0.187
  * **Accuracy**: \~94.69%
  * **Recall**: \~93.84%
  * **Precision**: \~95.88%
  * **AUC**: \~98.97%

The training history is also visualized with plots showing the model's loss and accuracy over epochs.

## 🌐 API Interface

An interactive interface is provided using the **Gradio** library, which allows for easy testing of the trained model. Users can upload their own images, and the application will return the predicted class and its confidence score. This provides a simple way to demonstrate the model's capability without needing to run the full notebook.
