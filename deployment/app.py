from flask import *
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

model = load_model('prediction_model')
cols = ['departments', 'bedrooms', 'bathrooms', 'property_type', 'operation_type', 'city']

@app.route('/')
def home():
    return render_template("home.html", )

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    print("***************",prediction)
    prediction = int(prediction.prediction_label[0])
    return render_template('home.html', pred='El precio de la propiedad es {}'.format(prediction) + " pesos")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(debug=True, host='0.0.0.0', port=port)
