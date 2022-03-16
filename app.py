from flask import Flask, render_template, request, jsonify
from flask_restx import Api,Resource

import tensorflow as tf
from keras.models import model_from_json
from keras.applications.mobilenet_v2 import preprocess_input

from PIL import Image
from io import BytesIO
from keras.preprocessing.image import img_to_array
import numpy as np

#app = Flask(__name__)


flask_app = Flask(__name__)      #creating flask WSGI Application
app = Api(app = flask_app,       #creating flask restplus api
		  version = "1.0", 
		  title = "Breast_Cancer Detection", 
		  description = "Breast_Cancer Detection")

name_space = app.namespace('prediction', description='Prediction APIs')


model = None
graph = tf.get_default_graph()


def load_request_image(image):
    image = Image.open(BytesIO(image))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((48, 48))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image

def load_model():
    json_file = open('./model/model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    global model
    model = model_from_json(model_json)
    model.load_weights("./model/weights.h5")

def predict_class(image_array):
    classes = ["Benign", "Malignant"]

    with graph.as_default():
        y_pred = model.predict(image_array, batch_size=None, verbose=0, steps=None)[0]
        class_index = np.argmax(y_pred, axis=0)
        confidence = y_pred[class_index]
        class_predicted = classes[class_index]
        return class_predicted, confidence

@name_space.route("/")
class MainClass(Resource):
    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    def post(self):
        try:
            image=request.files["image"].read()
            image=load_request_image(image)
            class_predicted,confidence=predict_class(image)
            image_class={"class":class_predicted,"confidence:":str(confidence)}
            response=jsonify(image_class)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "error": str(error)
            })

#if __name__ == "__main__":
 #   load_model()
  #  app.run(debug = False, threaded = False)

#if __name__ == "app":
 #   load_model()