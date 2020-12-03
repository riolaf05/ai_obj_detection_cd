import flask
from flask import Flask, jsonify, request, Response, flash, redirect
from flask_restful import Resource, Api, reqparse
import os
from datetime import *
import pytz
import json
from flask_cors import CORS, cross_origin
import base64
import re
import io
import logging
import configparser
import logging
import requests_oauthlib
from requests_oauthlib.compliance_fixes import facebook_compliance_fix
import hashlib
import requests
from werkzeug.utils import secure_filename
from decimal import Decimal
import tensorflow as tf
from PIL import Image
import numpy as np

UPLOAD_FOLDER = '/uploaded_photos'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
LABELS=['non incendio', 'incendio']

# This allows us to use a plain HTTP callback
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
api = Api(app)
CORS(app)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
def lite_model(images):
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="/model/lite_fire_detection_model.tflite")
  
  interpreter.allocate_tensors()
  interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
  interpreter.invoke()
  return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class uploadPhoto(Resource):
    @cross_origin()
    def post(self):

        if request.method == "POST":

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
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #image = np.array(Image.open(file))
                #lite_model(image) 
                #y_lite = np.argmax(probs_lite)
                #print(LABELS[y_lite])
                #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                status_code = Response(status=200)
                return "OK" #LABELS[y_lite] #render_template("/src/photo.html", user_image=full_filename)
        else:
            status_code = Response(status=400)
            return "KO" #render_template('index.html')

api.add_resource(uploadPhoto, '/uploadPhoto')

if __name__ == '__main__':
    app.run(app.run(debug=True, port=5000, host='0.0.0.0'))

