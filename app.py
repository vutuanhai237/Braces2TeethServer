
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from videoProcess import *
from detectMouth import detect
from util.visualizer import save_images
from util import html
from PIL import Image
from flask import Flask, flash, redirect, jsonify, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from os import path
from shutil import copyfile
import io, glob, base64, shutil, os.path, time
from flask_ngrok import run_with_ngrok
# Init server
app = Flask('braces2teeth')
app.config['UPLOAD_FOLDER'] = 'datasets'
CORS(app)
# run_with_ngrok(app)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

fileName = ''
def uploadFile():
    f = request.files['file']
    global fileName 
    fileName = f.filename[0:-4]
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True  
    opt.no_flip = True
    opt.display_id = -1
    return opt




@app.route('/process', methods=['POST'])
def processImage():
    # Pre-processing
    if (path.exists('results')):
        shutil.rmtree('results')
    if (path.exists('datasets')):
        shutil.rmtree('datasets')
    os.mkdir('datasets')
    opt = uploadFile()  
    dataset = create_dataset(opt)
    model = create_model(opt) 
    model.setup(opt)
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
    with open("results/braces2teeth/test_latest/images/" + fileName + "_fake.png", "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    return my_string


@app.route('/processvideo', methods=['POST'])
def processVideo():
    # preprocess
    if (path.exists('results')):
        shutil.rmtree('results', ignore_errors=True)
    if (path.exists('datasets')):
        shutil.rmtree('datasets', ignore_errors=True)
    os.makedirs('datasets/frames')
    opt = uploadFile()

    # extract video
    video = video2Images('datasets/' + fileName + '.mp4', 'datasets/frames')
    centeringAndSave('datasets/frames')
    resizeAllFile('datasets/frames')

    # process
    opt.dataroot = 'datasets/frames'
    dataset = create_dataset(opt)
    model = create_model(opt) 
    model.setup(opt)
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
    
    # concat each pair image
    concatPairImage('results/braces2teeth/test_latest/images/', 'results')
    images2Video('results/concat', 30, fileName)

    # not concat and image2video
    images2VideoNotConcat('results/not_concat', 30, fileName)
    return "Hello"

@app.route('/processvideo', methods=['GET'])
def getVideo():
    """Download a file."""
    return send_from_directory('', 'video.mp4', as_attachment=True)
@app.route('/processoriginvideo', methods=['GET'])
def getVideo2():
    """Download a file."""
    return send_from_directory('', 'videoNotConcat.mp4', as_attachment=True)
@app.route('/detectmouth', methods=['POST'])
def detectMouth():
    if (path.exists('datasets')):
        shutil.rmtree('datasets')
    os.mkdir('datasets')
    uploadFile()
    file = os.listdir('datasets')[0]
    frame = cv2.imread('datasets' + '/' + file)
    return detect(frame)

app.run()