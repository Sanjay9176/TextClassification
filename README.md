Overview
This project focuses on classifying movie reviews from the IMDB dataset as positive or negative using a deep learning model. The model is built using TensorFlow and leverages a pre-trained text embedding from TensorFlow Hub. The goal is to create a sentiment analysis model that learns from text data and predicts sentiment with good accuracy.

Features
Loads and processes the IMDB review dataset using TensorFlow Datasets

Uses a pre-trained embedding model (gnews-swivel-20dim) from TensorFlow Hub

Builds a neural network using Keras for binary classification

Trains and validates the model on real movie reviews

Evaluates performance on unseen test data

Installation
To run this project, make sure you have the required Python packages installed:

bash
Copy
Edit
pip install tensorflow tensorflow_hub tensorflow_datasets
Dataset Setup
The IMDB dataset is automatically downloaded using TensorFlow Datasets. The dataset includes 50,000 movie reviews labeled as either positive or negative.

The data is split as follows:

70% for training

30% for validation

A separate test set

How to Run
1. Import necessary libraries
python
Copy
Edit
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
2. Load and split the dataset
python
Copy
Edit
dataset_name = 'imdb_reviews'
train_split = 'train[:70%]'
validation_split = 'train[70%:]'
test_split = 'test'

train_data, validation_data, test_data = tfds.load(
    name=dataset_name,
    split=[train_split, validation_split, test_split],
    as_supervised=True)
3. Check for GPU availability
python
Copy
Edit
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "not available")
4. Use TensorFlow Hub for embedding
python
Copy
Edit
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
5. Build the neural network
python
Copy
Edit
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.summary()
6. Compile the model
python
Copy
Edit
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
7. Train the model
python
Copy
Edit
history = model.fit(
    train_data.shuffle(10000).batch(100),
    epochs=25,
    validation_data=validation_data.batch(100),
    verbose=1
)
8. Evaluate the model
python
Copy
Edit
results = model.evaluate(test_data.batch(100), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
Results
After training for 25 epochs, the model shows good performance on unseen test data. It effectively distinguishes between positive and negative reviews using the embedded text features. The training and validation accuracy improves consistently, indicating good generalization.

Author
[Sanjay Kumar Purohit]
