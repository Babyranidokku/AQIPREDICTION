from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json
    features = pd.DataFrame([data])
    
    
    scaled_features = scaler.transform(features)
    
    
    prediction = model.predict(scaled_features)
    
    
    bins = [0, 50, 100, 200, 300, 400, 500]
    labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    category = pd.cut(prediction, bins=bins, labels=labels)[0]

    return jsonify({'AQI': round(prediction[0], 2), 'Category': str(category)})

if __name__ == '__main__':
    app.run(debug=True)
