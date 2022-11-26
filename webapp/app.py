import logging
import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.utils.image_utils import img_to_array, load_img
from lime import lime_image
from tensorflow import keras
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)
app = Flask(__name__)

IMAGE_SIZE = (180, 180)
UPLOAD_FOLDER = 'uploads'
seed = 42
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('home.html', label='', imagesource='')

def preprocess(file_path):

    img = cv2.imread(file_path)

    img = img[...,::-1]

    return img, img[None,...]


def predict(model, img):
    predictions = model.predict(img)
    score = [float(x) for x in predictions[0]]

    return "This image is {:.0%} good, {:.0%} bad".format(score[0], score[1])


def evaluate(model, img, file_path):
    
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img,
        model.predict,
        top_labels=5,
        hide_color=None,
        num_samples=200
    )

    _, mask_1 = explanation.get_image_and_mask(
        explanation.top_labels[1],
        positive_only=True,
        num_features=500,
        hide_rest=True
    )

    # get the file from the folder
    img = cv2.imread(file_path)

    _, (ax1) = plt.subplots(1, 1, figsize=(100, 100))

    ax1.imshow(img, alpha=0.5)
    ax1.imshow(mask_1, alpha=0.5)
    ax1.axis('off')

    # upload new file to the uploads

    filename_explained = file_path.split('/')[-1]

    filename_explained = f'explained_{filename_explained}'

    filename_explained_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        filename_explained
    )

    plt.savefig(f'{filename_explained_path}', transparent=True)

    return f'{filename_explained_path}'


@app.route("/", methods=['POST', 'GET'])
def upload_file():

    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        print('file_path ', file_path)  # file_path  uploads/21.jpg
        file.save(file_path)

        loaded_model = tf.keras.models.load_model('/home/paulo/Documents/project-practical-mlops/circuit-board-ml/webapp/saved_model/')

        img_2d, img = preprocess(file_path)

        output_predict = predict(
            model=loaded_model,
            img=img
        )

        output_explanation_path = evaluate(
            model=loaded_model,
            img=img_2d,
            file_path=file_path
        )

        return render_template(
            "home.html",
            label=output_predict,
            imagesource=output_explanation_path
        )


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, threaded=False, port=port, host='0.0.0.0')
    # app.run(debug=True, threaded=False)
