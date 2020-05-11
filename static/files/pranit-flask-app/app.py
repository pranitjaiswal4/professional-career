import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

app = Flask(__name__) 
model = pickle.load(open('model.pkl', 'rb'))
tfidfconverter = pickle.load(open('tfidfconverter1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    mcom = {'comment': [request.form['review']]}
    mdf = pd.DataFrame(mcom, columns = ['comment'])
    X_single = tfidfconverter.transform(mdf['comment'])
    prediction = model.predict(X_single)

    return render_template('index.html', prediction_text=f'Predict rating: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
