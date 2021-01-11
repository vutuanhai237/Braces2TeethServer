
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image
from flask import Flask, flash, redirect, jsonify, request, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from os import path
from shutil import copyfile
import io
import base64
import shutil
import os.path
import cv2
# Init server
app = Flask('braces2teeth')
app.config['UPLOAD_FOLDER'] = 'datasets\\test'
CORS(app)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

fileName = ''
def uploadImage():
    f = request.files['file']
    global fileName 
    fileName = f.filename[0:-4]
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    return


@app.route('/process', methods=['POST'])
def predict():
    # Pre-processing
    if (path.exists('results')):
        shutil.rmtree('results')
    if (path.exists('datasets\\test')):
        shutil.rmtree('datasets\\test')
    os.mkdir(os.path.join('datasets', 'test'))
    uploadImage()
    # copyfile(request.form.get('file'), 'datasets\\test\\img.png')
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
    with open("results\\braces2teeth\\test_latest\\images\\" + fileName + "_fake.png", "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    return my_string
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')