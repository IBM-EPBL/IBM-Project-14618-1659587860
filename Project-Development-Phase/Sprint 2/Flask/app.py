from flask import Flask,render_template,request
import numpy as np
import os
import operator
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder="templates")
model=load_model('../Model Building/gesture.h5')
print("Model is loaded from local system")

@app.route("/")
def root():
	return render_template("home.html")

@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/intro")
def intro():
	return render_template("intro.html")

@app.route("/launch")
def launch():
	return render_template("launch.html")


@app.route('/index',methods=['GET','POST'])
def index():
    return render_template("lauch.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    #Getting input and storing it
    if request.method == 'POST':
        print('inside launch function')
        f=request.files['image']

        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        print('img saved successfully')
        print(file_path)
        # test_image=cv2.imread(file_path,cv2.IMREAD_COLOR)
        # test_image=cv2.resize(test_image,(64,64))
        # result= model.predict(test_image.reshape(1,64,64,1))
        img = load_img(file_path, grayscale=True, target_size=(64, 64))
        x = img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        result = model.predict(x)
        print(result)
        prediction = {'ZERO':result[0][0],'ONE':result[0][1],'TWO':result[0][2],'THREE':result[0][3],'FOUR':result[0][4],'FIVE':result[0][5]}
        prediction=sorted(prediction.items(),key=operator.itemgetter(1),reverse=True)

    return render_template("home.html")

if __name__ == "__main__":
	app.run(debug=True)
