import time
import os
import cv2
import numpy as np
import cnn_svm, cnn_svm_2, cnn_svm_3
from PIL import Image
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dir1 = ['static/image/training/acral melanoma/',
        'static/image/testing/acral melanoma/',
        'static/image/training/benign nevi/',
        'static/image/testing/benign nevi/']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prediksi(model, data):
    feats, names = model.ekstrak_train()
    feats_test, names_test = model.ekstrak_test(data)
    x_train = feats
    x_test= feats_test

    lb = LabelEncoder()
    y_train = lb.fit_transform(names)
    y_test = lb.fit_transform(names_test)

    svmclf = SVC(C=100, gamma=0.1, kernel='linear', probability=True)
    modelsvm = svmclf.fit(x_train, y_train)
    start = time.time()
    y_testSVM = modelsvm.predict(x_test)
    labels = (y_testSVM > 0.5).astype(int)
    runtimes = round(time.time()-start,4)
    proba = modelsvm.predict_proba(x_test)
    proba = max(proba[0]) * 100

    return labels, runtimes, proba

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('/select.html', )

@app.route('/', methods=['GET', 'POST'])
def result_file():
    return render_template('/select.html')

@app.route('/predict-upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename("temp.jpg")
            data = os.path.join('static/uploads/', 'temp.jpg')
            file.save(data)
            chosen_model = request.form['select_model_upload']
            if chosen_model == 'model1':
                model1=cnn_svm
                labels, runtimes, proba = prediksi(model1, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            elif chosen_model == 'model2':
                model2=cnn_svm_2
                labels, runtimes, proba = prediksi(model2, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            elif chosen_model == 'model3':
                model3=cnn_svm_3
                labels, runtimes, proba = prediksi(model3, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return predict_result(chosen_model, runtimes, proba, '/uploads/temp.jpg', labels)
    return render_template('index.html')

@app.route('/predict-sample', methods=['POST'])
def predict():
    if request.method == 'POST':
        filename = request.form.get('input_image')
        img = Image.open(filename)
        img = img.save("static/uploads/temp.jpg")
        data = os.path.join('static/uploads/', 'temp.jpg')
        chosen_model = request.form['select_model_sample']

        if chosen_model == 'model1':
            model1=cnn_svm
            labels, runtimes, proba = prediksi(model1, data)
        elif chosen_model == 'model2':
            model2=cnn_svm_2
            labels, runtimes, proba = prediksi(model2, data)
        elif chosen_model == 'model3':
            model3=cnn_svm_3
            labels, runtimes, proba = prediksi(model3, data)

    return predict_result(chosen_model, runtimes, proba, '/uploads/temp.jpg', labels)

def predict_result(model, run_time, probs, img, labels):
    class_list = ['acral melanoma', 'benign nevi']
    if labels == 0:
        labels = class_list[0]
    else:
        labels = class_list[1]
    return render_template('/result_select.html', labels=labels, 
                            probs=probs, model=model,
                            run_time=run_time, img=img)

if __name__ == "__main__": 
        app.run(debug=True)