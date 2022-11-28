import logging
import os
from pathlib import Path

import cv2

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, Path, Request
from fastapi.responses import JSONResponse
import cv2

log = logging.getLogger(__name__)

app = FastAPI(
    title='SMART Data Science Application',
    description='A Smart Data Science Application running on FastAPI + uvicorn',
    version='0.0.1'
)

loaded_model = tf.keras.models.load_model('./saved_model/model.h5')
dictionary = {0: 'adequate', 1: 'inadequate'}


@app.post("/predict/file_path")
def main(file_path):

    files = os.listdir(file_path)
    result_dict = {}

    for file in files[:200]:

        img = cv2.imread(os.path.join(file_path, file))
        img = img[..., ::-1]

        df_output = loaded_model.predict(img[None, ...], batch_size=16, verbose=2, steps=None)

        result = dictionary[np.argmax(df_output[0])]

        result_dict[file] = result

    return result_dict


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
