import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, Path, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

app = FastAPI(
    title='Circuit board classifier APP',
    description='A quality check App running on FastAPI + uvicorn',
    version='0.0.1')

loaded_model = tf.keras.models.load_model('./saved_model/model.h5')

dictionary = {0: 'adequate', 1: 'inadequate'}


@app.post("/predict/file_path")
def main(file_path):

    files = os.listdir(file_path)
    result_dict = {}

    for file in files[:10]:

        img = cv2.imread(os.path.join(file_path, file))
        img = img[..., ::-1]

        df_output = loaded_model.predict(img[None, ...], batch_size=1, verbose=2, steps=1)

        result = dictionary[np.argmax(df_output[0])]

        result_dict[file] = result

    # TODO write json in a s3

    json_result = jsonable_encoder(result_dict)
    out_file = open("result.json", "w", encoding='utf8')
    json.dump(json_result, out_file, indent=2)
    out_file.close()

    return JSONResponse(result_dict)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
