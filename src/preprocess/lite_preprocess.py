"""Lite Preprocesser."""
from preprocess.preprocess import Preprocess
import numpy as np
import pandas as pd
import os, random
import tensorflow as tf
from tensorflow import keras
from keras import layers, preprocessing, models, utils, applications, layers
from tensorflow.python.client import device_lib
import glob
import shutil
from sklearn.model_selection import train_test_split
from input.config.base_config import Config

BATCH_SIZE=15
SEED = 42
FAST_RUN = True
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


class LitePreprocess(Preprocess):
    """Preprocess class."""

    def __init__(
        self,
        config
    ):
        """Definition of Preprocess constructor.

        Args:
            
        """
        super().__init__(config)

        self.config = config
    
    def get_data(self):

        # listing all files
        files_path = []
        for path in glob.glob(self.config.downloaded_data_path+'/*/*/*.jpg'):
            files_path.append(path)

        # downloaded -> organized
        for files in files_path:
            shutil.move(src=files, dst=self.config.organized_data_path)

        # organized -> shuffle files -> train and validation
        organized_files = os.listdir(self.config.organized_data_path)
        organized_files = random.choices(organized_files, k = len(organized_files))

        self.config.logger.info(f'Done Getting data')

        return organized_files

    def quality_check(self):
    
        to_delete = True
        if to_delete:

            num_skipped = 0
            
            for fname in os.listdir(self.config.organized_data_path):
                
                fpath = os.path.join(self.config.organized_data_path, fname)
                try:
                    fobj = open(fpath, "rb")
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
                finally:
                    fobj.close()

                if not is_jfif:
                    num_skipped += 1
                    # Delete corrupted image
                    os.remove(fpath)

        self.config.logger.info(f'Deleted {num_skipped} images')

    def create_dataset(self, organized_files):
        labels = []

        for filename in organized_files:
            label = filename.split('_')[1].split('.')[0]
            if label == 'temp':
                labels.append(0)
            else:
                labels.append(1)

        df = pd.DataFrame({
            'filename': organized_files,
            'label': labels
        })

        df["label"] = df["label"].replace({0: 'Good', 1: 'Bad'})

        return df

    def train_test_split(self, df):

        train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)

        return train_df, validate_df

    def image_generator(self, train_df, validate_df):

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

        train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, 
        directory=self.config.organized_data_path, 
        x_col='filename',
        y_col='label',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        seed=SEED
        )

        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        validation_generator = (
            validation_datagen.flow_from_dataframe(
                dataframe=validate_df, 
                directory = self.config.organized_data_path, 
                x_col='filename',
                y_col='label',
                target_size=IMAGE_SIZE,
                class_mode='categorical',
                batch_size=BATCH_SIZE,
                seed=SEED
                )
        )

        return train_generator, validation_generator

    def run(self):

        organized_files = self.get_data()

        self.quality_check()

        df = self.create_dataset(organized_files)

        train_df, validate_df = self.train_test_split(df)

        train_generator, validation_generator = self.image_generator(train_df, validate_df)

        return train_generator, validation_generator





        
        

    

