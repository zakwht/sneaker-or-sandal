from flask import Flask, request, flash, redirect, url_for, render_template
import os
from io import BytesIO
from base64 import b64encode
from werkzeug.utils import secure_filename
from joblib import dump, load
from fashionmnist.utils import mnist_reader
from sklearn.svm import SVC
from numpy import array, reshape
from PIL import Image

SEED = 1234

def clean(data, end=None):
  X, y = data
  # find indices of sandals and sneakers (class 5/7)
  indices = [i for i, x in enumerate(y) if x == 5 or x == 7]
  X = [X[i] for i in indices]
  # convert y to binary values
  y = [0 if y[i] == 5 else 1 for i in indices]
  # preprocess by scaling X
  X = list(array(X)/255)
  return X[:end], y[:end]

def gausssvm(train, C, gamma):
  return SVC(random_state=SEED, max_iter=-1, C=C, gamma=gamma).fit(*train)

def initclf():
  train = clean(mnist_reader.load_mnist('fashionmnist/data/fashion', kind='train'))
  test = clean(mnist_reader.load_mnist('fashionmnist/data/fashion', kind='t10k'))
  clf = gausssvm(train, 4, 0.036) # best model as concluded from analysis
  dump(clf, 'model.joblib')

def cleanFile(file):
  base = Image.open(file).convert('RGBA')
  base = base.resize((28, 28), 1)
  img = Image.new('RGB', (28, 28))
  img.paste(base, (0,0), base)
  return img.copy().convert('L')

def scoreFile(clf, file):
  img = cleanFile(file)
  arr = array(img).flatten() / 255
  return "sandal" if clf.predict([arr]) == 0 else "sneaker"

clf = load("model.joblib")

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = '/uploads'

baseHTML = '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    # print(request.files['file'].filename)
    file = request.files['file']
    # print('data:image/png;base64,' + b64encode(file.read()).decode())
    if file and allowed_file(file.filename):
      url = 'data:image/png;base64,' + b64encode(file.read()).decode()
      buff = BytesIO()
      cleanFile(file).save(buff, "PNG")
      url2 = 'data:image/png;base64,' + b64encode(buff.getvalue()).decode()
      filename = secure_filename(file.filename)
      return render_template('template.html', file_url=url, cleaned_file_url=url2, type=scoreFile(clf, file))
      return (
        baseHTML 
        + "<br/>this looks like a <b>" 
        + scoreFile(clf, file) 
        + "</b> to me<br/>"
        + "<img src=" + url + " width=200px>"
        + "<img src=" + url2 + " width=200px>"
        )
  return render_template('template.html')
