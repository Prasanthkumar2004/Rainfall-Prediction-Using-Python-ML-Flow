import os
from flask import Flask, request, render_template
import pandas as pd
import mlflow.sklearn

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.debug = True  # Enable debugging

# Load MLFlow model
mlflow.set_experiment("Rainfall")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

model_version = 1
production_model_name = "rainfall-prediction-production"
prod_model_uri = f"models:/{production_model_name}@champion"

# Load the model
try:
    loaded_model = mlflow.sklearn.load_model(prod_model_uri)
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

# Feature names
feature_names = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = [
            float(request.form['pressure']),
            float(request.form['dewpoint']),
            float(request.form['humidity']),
            float(request.form['cloud']),
            float(request.form['sunshine']),
            float(request.form['winddirection']),
            float(request.form['windspeed']),
        ]

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Ensure model is loaded
        if loaded_model is None:
            return render_template('index.html', prediction_result="Error: Model not loaded!")

        # Make prediction
        prediction = loaded_model.predict(input_df)

        # Define Prediction Categories
        if prediction[0] == 0:
            result = "No Rainfall Expected"
        else:
            if input_data[0] < 1000 and input_data[2] > 80:
                result = "Heavy Rainfall Expected (20mm+)"
            elif input_data[0] < 1010 and input_data[2] > 60:
                result = "Light Drizzle Expected (0.5mm - 2mm)"
            elif input_data[0] < 1010 and input_data[3] > 80:
                result = "Thunderstorms Expected (30mm+)"
            else:
                result = "Light Rain Expected (3mm - 7mm)"

        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Run Flask's built-in development server
