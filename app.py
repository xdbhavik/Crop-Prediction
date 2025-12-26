import numpy as np
from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)

# Load your trained model
# Ensure model.pkl is in the same directory as app.py
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Convert form inputs to a list of floats
    # This pulls Nitrogen, Phosphorus, Potassium, etc., in the order they appear in the HTML
    float_features = [float(x) for x in request.form.values()]
    
    # Convert to a 2D array as expected by Scikit-Learn
    features = [np.array(float_features)]
    
    # Get prediction from model
    prediction = model.predict(features)
    
    # Extract the first element from the prediction array/list
    result = prediction[0]
    
    return render_template(
        "index.html", 
        prediction_text="Recommended Crop: {}".format(result)
    )

if __name__ == "__main__":
    flask_app.run(debug=True)
    