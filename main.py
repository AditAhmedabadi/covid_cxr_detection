from flask import Flask, render_template, request,  redirect, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

UPLOAD_FOLDER = './assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__, static_url_path='/assets',
            static_folder='./assets')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@ app.route('/')
def root():
    return render_template('index.html')


@ app.route('/index.html')
def index():
    return render_template('index.html')


@ app.route('/about.html')
def about():
    return render_template('about.html')


@ app.route('/covid.html')
def upload():
    return render_template('covid.html')


@ app.route("/show", methods=['POST', 'GET'])
def uploaded_chest():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

        xception_chest = load_model('./models/my_model.h5')

    img_arr = preprocess_test_img('./assets/images/upload_chest.jpg')
    xception_pred = xception_chest.predict(img_arr)
    probability = xception_pred[0][0]
    print("Xception Predictions:")
    if probability > 0.5:
        xception_chest_pred = ('The patient has ' + str(
            '%.2f' % (probability*100) + ' % COVID'))
    else:
        xception_chest_pred = ('The patient is ' + str(
            '%.2f' % ((1-probability)*100) + ' % Non-COVID'))
    print(xception_chest_pred)

    return render_template('show.html', xception_chest_pred=xception_chest_pred)


def preprocess_test_img(i):
    pil_img = Image.open(i)
    img_arr = np.asarray(pil_img).astype(np.float32)
    if len(img_arr.shape) > 2 and img_arr.shape[2] == 4:
        # convert the image from RGBA2RGB
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)

        img_arr = tf.convert_to_tensor(img_arr, dtype=tf.float32)
    if len(img_arr.shape) == 2:
        img_arr = tf.expand_dims(img_arr, axis=-1)
        img_arr = tf.image.grayscale_to_rgb(
            img_arr)  # convert grayscale to rgb
    img_arr = tf.image.resize(img_arr, (299, 299))
    img_arr = tf.expand_dims(img_arr, axis=0)
    return img_arr


if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)
