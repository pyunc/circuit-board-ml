"""App Train."""
import glob
import os
import random
import shutil
import warnings
import joblib
warnings.simplefilter("ignore") # disable Keras warnings for this tutorial

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import applications, layers, models, preprocessing, utils
from tensorflow import keras
from tensorflow.python.client import device_lib

from input.config.base_config import Config
from models.architecture.architecture_definition import ArchsList
from preprocess.preprocess_definition import PreprocessingList

config = Config()

FAST_RUN = True

# input
def input(config):

    preprocess = PreprocessingList[config.preprocess].value

    train_generator, validation_generator = preprocess(config).run()

    return train_generator, validation_generator

def build_model(config):

    selected_model = ArchsList[config.selected_model].value

    model, callbacks = selected_model(config).build()

    return model, callbacks

def main(config):

    train_generator, validation_generator = input(config)

    model, callbacks = build_model(config)

    batch_size = 16
    
    epochs=1 if FAST_RUN else 50

    history = model.fit_generator(
        generator=train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=int(validation_generator.n)//batch_size,
        steps_per_epoch=int(train_generator.n)//batch_size,
        callbacks=callbacks
    )

    filename = 'model.h5'
    filepath = '/home/paulo/Documents/project-practical-mlops/circuit-board-ml/src/input/saved_model'

    with open(filepath+'/'+filename, 'wb') as f:
        model.save_weights(filepath+'/'+'model.h5')
    
if __name__ == '__main__':
    main(config)

# # evaluation

# epochs = 15

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
# ax1.plot(history.history['loss'], color='b', label="Training loss")
# ax1.plot(history.history['val_loss'], color='r', label="validation loss")
# ax1.set_xticks(np.arange(1, epochs, 1))
# ax1.set_yticks(np.arange(0, 1, 0.1))

# ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
# ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
# ax2.set_xticks(np.arange(1, epochs, 1))

# legend = plt.legend(loc='best', shadow=True)
# plt.tight_layout()
# plt.show()