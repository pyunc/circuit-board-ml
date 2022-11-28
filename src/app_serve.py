import os
import warnings

import cv2
import numpy as np
import tensorflow as tf

from input.config.base_config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings('ignore')

config = Config()


def load_model():
    '''

    '''
    try:
        config.logger.info("Loading model artifact... {}".format(config.model_save_dev_path))
        # model_file = os.path.join(config.model_save_path, 'model.h5')
        model: object = tf.keras.models.load_model(
            os.path.join(
                config.model_save_dev_path,
                'model.h5'
                )
            )
        config.logger.info("Loaded model!")
        return model
    except Exception as e:
        config.logger.error(f'failed to load model due to {e}')


def predict():
    """_summary_

    Returns:
        _type_: _description_
    """

    # TODO batch classification

    config.logger.info(f'Model selected for inference :{config.selected_model}')

    model = load_model()
    img = cv2.imread(r'/home/paulo/Documents/circuit-board-ml/src/input/data/organized_data/00041000_test.jpg')
    img = img[..., ::-1]

    # predict
    df_output = model.predict(img[None, ...], batch_size=None, verbose=2, steps=None)

    dictionary = {0: 'adequate', 1: 'inadequate'}

    config.logger.info(dictionary[np.argmax(df_output[0])])

    return dictionary[np.argmax(df_output[0])]


if __name__ == '__main__':
    predict()
