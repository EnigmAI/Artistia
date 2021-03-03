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
import Sketching
import Pixelate
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
    if os.path.exists("static/uploads/source1.png"):
        os.remove("static/uploads/source1.png")
    if os.path.exists("static/uploads/source2.png"):
        os.remove("static/uploads/source2.png")
    if os.path.exists("static/uploads/source3.png"):
        os.remove("static/uploads/source3.png")
    if os.path.exists("static/uploads/style.png"):
        os.remove("static/uploads/style.png")
    if os.path.exists("static/results/result.png"):
        os.remove("static/results/result.png")
    if os.path.exists("static/results/result_color.png"):
        os.remove("static/results/result_color.png")
    if os.path.exists("static/results/result_sketch.png"):
        os.remove("static/results/result_sketch.png")
    if os.path.exists("static/results/result_pixel.png"):
        os.remove("static/results/result_pixel.png")
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
        print(h, w)
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
                app.config['UPLOAD_FOLDER'], 'source1.png'))
        Colorization.color()
        render_template('./colorization.HTML')
    return render_template('./colorization.HTML')

@app.route('/sketch', methods=['GET', 'POST'])
def get_img_sketch():
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
                app.config['UPLOAD_FOLDER'], 'source2.png'))
        Sketching.sketch()
        render_template('./sketching.HTML')
    return render_template('./sketching.HTML')

@app.route('/pixel', methods=['GET', 'POST'])
def get_img_pixel():
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
                app.config['UPLOAD_FOLDER'], 'source3.png'))
        if 'size' not in request.form:
            Pixelate.pixelate()
            render_template('./pixel.html')
            return render_template('./pixel.html')
        size = request.form['size']
        Pixelate.pixelate(pixel_size = int(size))
        render_template('./pixel.html')
    return render_template('./pixel.html')

app.run(port=5000, debug=True)
