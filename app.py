from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model from the .pkl file
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = []  # Store user input features here

    # Retrieve user inputs from the form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])

    # Append user inputs to the features list
    features.append([feature1, feature2, feature3, feature4])

    # Make prediction using the loaded model
    predicted_species = model.predict(features)
    predicted_species_name = predicted_species[0]
    
    return render_template('result.html', species=predicted_species_name)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
