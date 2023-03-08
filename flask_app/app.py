import os
import numpy as np
import pandas as pd
import pickle

import cv2
import keras
from keras.models import Model
from keras.utils import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import pyttsx3

from flask import Flask, render_template, request, jsonify



UPLOAD_FOLDER = 'C:/Users/sadish/OneDrive/Desktop/Image_Text_Speech_Generation-master/flask_app/static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = InceptionV3(weights='imagenet', include_top=True)
model_new = Model(model.input, model.layers[-2].output)

def get_inceptionv3_feature(image):
    cv_img = cv2.imread(image)
    img_resize = cv2.resize(cv_img, (299,299))
    pre_processed_image = preprocess_input(img_resize)
    pre_processed_image = np.resize(pre_processed_image,(1,299,299,3))
    
    image_feature = model_new.predict(pre_processed_image)
    
    return image_feature

def get_word(prediction, tokenizer):
    word = tokenizer.index_word[prediction]
    return word

def get_description(model, photo_feature, tokenizer, max_word_len=34):
    desc =  'startseq'
    for i in range(max_word_len):
        word_vector = tokenizer.texts_to_sequences([desc])[0]
        word_vector = pad_sequences([word_vector], maxlen=max_word_len)#, dtype='int32', padding='post', truncating='post', value=0.0)#[0]
        prediction = model.predict([photo_feature.reshape(1,2048), word_vector], verbose=0)
        prediction = np.argmax(prediction)
        word = get_word(prediction, tokenizer)
        desc += " "+str(word)
        if word == 'endseq':
            break
        i += 1
        
    return desc

@app.route("/", methods=["GET", "POST"])
def index():
    with open(r'C:\Users\sadish\OneDrive\Desktop\Image_Text_Speech_Generation-master\flask_app\ml\tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = keras.models.load_model(r'C:\Users\sadish\OneDrive\Desktop\Image_Text_Speech_Generation-master\flask_app\ml\modelLSTM_19.h5')

    image_name = ''
    description = ''

    if request.method == "POST":
        image = request.files['file']
        image_name = image.filename
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        image.save(image_path)
        
        image_feature = get_inceptionv3_feature(image_path)
        description = get_description(model, image_feature, tokenizer)

        description = " ".join(description.split(" ")[1:-1])

    print(description)
   
    engine = pyttsx3.init()
    engine.say(description)
    engine.runAndWait()

    return render_template('index.html', jinja_image=image_name, description= description) 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
