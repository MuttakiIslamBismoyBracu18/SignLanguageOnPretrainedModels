from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed
import random
seed(2)
import tensorflow as tf
tf.random.set_seed(2)

import cv2
from glob import glob

from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from pathlib import Path
import pandas as pd
import numpy as np
import time
import itertools

# Ensure the GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Set directories
TRAIN_DIR = r"E:\Undergrad_Research\DATASET_Final\Train"
TEST_DIR = r"E:\Undergrad_Research\DATASET_Final\Test"
MODEL_DIR = r"E:\Undergrad_Research\DATASET_Final"

classes = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]
classes.sort()

target_size = (128, 128)
target_dims = (128, 128, 3)
N_classes = 36
validation_split = 0.1
batch_size = 64

MODEL_PATH = MODEL_DIR + '/cnn-model.h5'
MODEL_WEIGHTS_PATH = MODEL_DIR + '/cnn-model-weights.h5'

print(f'Save model to disk? Yes')

# Data Augmentation for Image Generation
def preprocess_image(image):
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobely

def make_generator(options):
    validation_split = options.get('validation_split', 0.0)
    preprocessor = options.get('preprocessor', None)
    data_dir = options.get('data_dir', TRAIN_DIR)

    augmentor_options = {
        'samplewise_center': True,
        'samplewise_std_normalization': True,
    }
    if validation_split is not None:
        augmentor_options['validation_split'] = validation_split

    if preprocessor is not None:
        augmentor_options['preprocessing_function'] = preprocessor

    flow_options = {
        'target_size': target_size,
        'batch_size': batch_size,
        'shuffle': options.get('shuffle', None),
        'subset': options.get('subset', None),
    }

    data_augmentor = ImageDataGenerator(**augmentor_options)
    return data_augmentor.flow_from_directory(data_dir, **flow_options)

# Load model from disk if it exists
def load_model_from_disk():
    model_file = Path(MODEL_PATH)
    model_weights_file = Path(MODEL_WEIGHTS_PATH)

    if model_file.is_file() and model_weights_file.is_file():
        print('Retrieving model from disk...')
        model = load_model(model_file.__str__())

        print('Loading CNN model weights from disk...')
        model.load_weights(model_weights_file)
        return model

    return None

# Try to load the existing model
CNN_MODEL = load_model_from_disk()
REPROCESS_MODEL = (CNN_MODEL is None)

print(f'Need to reprocess? {REPROCESS_MODEL}')

# Build the CNN model
def build_model(save=False):
    print('Building model afresh...')

    model = Sequential()

    model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
    model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(N_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if save:
        model.save(MODEL_PATH)

    return model

if CNN_MODEL is None:
    CNN_MODEL = build_model(save=True)

# Function to generate data for training and validation
def make_generator_for(subset):
    generator_options = dict(
        validation_split=validation_split,
        shuffle=True,
        subset=subset,
        preprocessor=preprocess_image,
    )
    return make_generator(generator_options)

# Fit the model on training data
def fit_model(model, train_generator, val_generator, save=False):
    history = model.fit(train_generator, epochs=32, validation_data=val_generator)

    if save:
        model.save_weights(MODEL_WEIGHTS_PATH)

    return history

# Create generators for training and validation
CNN_TRAIN_GENERATOR = make_generator_for('training')
CNN_VAL_GENERATOR = make_generator_for('validation')

HISTORY = None
if REPROCESS_MODEL:
    start_time = time.time()
    HISTORY = fit_model(CNN_MODEL, CNN_TRAIN_GENERATOR, CNN_VAL_GENERATOR, save=True)
    print(f'Fitting the model took ~{time.time() - start_time:.0f} second(s).')

# Evaluation and visualization functions
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.cividis):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

def plot_confusion_matrix_with_default_options(y_pred, y_true, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 16))
    plot_confusion_matrix(cm, classes)
    plt.show()

# Model evaluation and prediction
def evaluate_model(generator):
    evaluations = CNN_MODEL.evaluate(generator)
    predictions = CNN_MODEL.predict(generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes
    return dict(y_pred=y_pred, y_true=y_true)

# Evaluate the validation dataset
CNN_VALIDATION_SET_EVAL = evaluate_model(CNN_VAL_GENERATOR)
print(classification_report(**CNN_VALIDATION_SET_EVAL, target_names=classes))
plot_confusion_matrix_with_default_options(**CNN_VALIDATION_SET_EVAL, classes=classes)

# Evaluate the test dataset
test_generator_options = dict(
    validation_split=0.0,
    data_dir=TEST_DIR,
    shuffle=False,
    preprocessor=preprocess_image,
)
test_generator = make_generator(test_generator_options)

CNN_TEST_SET_EVAL = evaluate_model(test_generator)
print(classification_report(**CNN_TEST_SET_EVAL, target_names=classes))
plot_confusion_matrix_with_default_options(**CNN_TEST_SET_EVAL, classes=classes)

# Plot training and validation accuracy and loss
plt.plot(HISTORY.history['accuracy'])
plt.plot(HISTORY.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc='lower right')
plt.show()

plt.plot(HISTORY.history['loss'])
plt.plot(HISTORY.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc='upper right')
plt.show()
