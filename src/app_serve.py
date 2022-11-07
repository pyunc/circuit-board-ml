import os

import cv2
import tensorflow as tf

from input.config.base_config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings('ignore')

config = Config()

def load_model():
    '''

    '''
    try:
        config.logger.info("Loading model artifact... {}".format(config.model_save_path))
        # model_file = os.path.join(config.model_save_path, 'model.h5')
        model: object = tf.keras.models.load_model(config.model_save_path)
        config.logger.info("Loaded model!")
        return model
    except Exception as e:
        config.logger.error(f'failed to load model due to {e}')

# def predict(df_input, model):
def predict():
    """_summary_

    Returns:
        _type_: _description_
    """

    # config.logger.info('Dataset shape: {} samples and {} features.'.format(*df_input.shape))

    
    config.logger.info(f'Model selected for inference :{config.selected_model}')

    model = load_model()
    img = cv2.imread(r'/home/paulo/Documents/project-practical-mlops/circuit-board-ml/src/input/data/organized_data/00041000_test.jpg')
    img = img[...,::-1]                                                                 

    # predict
    df_output = model.predict(img[None,...], batch_size=None, verbose=2, steps=1)

    config.logger.info(f'Model predicted :{df_output}')

    return df_output

if __name__ == '__main__':
    predict()