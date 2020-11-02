import os
from urllib.parse import urlparse
from uuid import uuid4
import requests
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import HighResST
import LowResST
import Colorization

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
sourcename = ''
stylename = ''


@app.route('/')
def index():
    if os.path.exists("static/uploads/source.png"):
        os.remove("static/uploads/source.png")
    if os.path.exists("static/uploads/style.png"):
        os.remove("static/uploads/style.png")
    if os.path.exists("static/results/result.png"):
        os.remove("static/results/result.png")
    if os.path.exists("static/results/result_color.png"):
        os.remove("static/results/result_color.png")
    return render_template('./index.HTML')


@app.route('/stylize', methods=['GET', 'POST'])
def get_img():
    if request.method == 'POST':
        if 'source' not in request.files:
            print('No file part')
            return redirect(request.url)
        if 'style' not in request.files:
            print('No file part')
            return redirect(request.url)
        source = request.files['source']
        style = request.files['style']
        if style.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if source.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if source:
            global sourcename
            sourcename, extension1 = os.path.splitext(source.filename)
            source.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'source.png'))
        if style:
            global stylename
            stylename, extension2 = os.path.splitext(style.filename)
            style.save(os.path.join(app.config['UPLOAD_FOLDER'], 'style.png'))
        content_path = 'static/uploads/source.png'
        x = LowResST.load_img_and_preprocess(content_path)
        h, w = x.shape[1:3]
        if h > 400 or w > 400:
            tf.compat.v1.enable_eager_execution()
            HighResST.styleTransfer(extension1, extension2)
        else:
            tf.compat.v1.disable_eager_execution()
            LowResST.styleTransfer(extension1, extension2)
        render_template('./styletransfer.HTML')
    return render_template('./styletransfer.HTML')


@app.route('/colorize', methods=['GET', 'POST'])
def get_img_color():
    if request.method == 'POST':
        if 'source' not in request.files:
            print('No file part')
            return redirect(request.url)
        source = request.files['source']
        if source.filename == '':
            print('No selected file')
            return redirect(request.url)
        if source:
            global sourcename
            sourcename, extension1 = os.path.splitext(source.filename)
            source.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'source.png'))
        Colorization.color()
        render_template('./colorization.HTML')
    return render_template('./colorization.HTML')


app.run(port=5000)
