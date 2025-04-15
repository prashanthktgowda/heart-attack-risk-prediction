from flask import *
import os
from werkzeug.utils import secure_filename
import label_image

import image_fuzzy_clustering as fem
import os
import secrets
from PIL import Image
from flask import url_for, current_app



def load_image(image):
    text = label_image.main(image)
    return text




def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image




app = Flask(__name__)
model = None

UPLOAD_FOLDER = os.path.join(app.root_path ,'static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
@app.route('/first')
def first():
    return render_template('index.html')

 
  
    
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/upload')
def upload():
    return render_template('index1.html')

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        i=request.form.get('cluster')
        f = request.files['file']
        fname, f_ext = os.path.splitext(f.filename)
        original_pic_path=save_img(f, f.filename)
        destname = 'em_img.jpg'
        fem.plot_cluster_img(original_pic_path,i)
    return render_template('success.html')

def save_img(img, filename):
    picture_path = os.path.join(current_app.root_path, 'static/images', filename)
    # output_size = (300, 300)
    i = Image.open(img)
    # i.thumbnail(output_size)
    i.save(picture_path)

    return picture_path



@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make predictions
        result = load_image(file_path)
        result = result.title()
        d = {"1":" → Age = 30-35 , SBP = 140-160 mmHg, DBP = 80-90 mmHg, BMI = 27-29, HbA1c = 4-5.6, Risk of Heart Attack = Very Low Risk 20% ",
	'2':" → Age = 35-40 , SBP = 150-166 mmHg, DBP = 85-95 mmHg, BMI = 29-31, HbA1c = 7.5 -10.5 , Risk of Heart Attack = Mild Risk 40%",
        '3':" → Age = 35-40 , SBP = 120-136 mmHg, DBP = 75-55 mmHg, BMI = 18-25, HbA1c = 5.5 -6.5 , Risk of Heart Attack = No Risk You are Healthy",
        '4':" → Age = 45-60 , SBP = 160-176 mmHg, DBP = 95-100 mmHg, BMI = 30-35, HbA1c = 13.4 -14.9 , Risk of Heart Attack = High Chance of Heart Attack 60%",
        "0":" → Age = 20-25 , SBP = 111-126 mmHg, DBP = 80-85 mmHg, BMI = 18-25, HbA1c = 5.4 -7.0 , Risk of Heart Attack = No Risk You are Healthy",
        "5":" → invalid input " }
        result = result+d[result]
        #result2 = result+d[result]
        #result = [result]
        #result3 = d[result]        
        print(result)
        #print(result3)
        os.remove(file_path)
        return result
        #return result3
    return None

if __name__ == '__main__':
    app.run()