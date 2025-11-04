from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs
        age = float(request.form["feature1"])
        time_spent = float(request.form["feature2"])
        interest_score = float(request.form["feature3"])
        ad_duration = float(request.form["feature4"])

        # Prepare data
        features = np.array([[age, time_spent, interest_score, ad_duration]])
        prediction = model.predict(features)[0]

        # User-friendly output
        if prediction == 1:
            message = "✅ Recommended Ad Type: User Likely to Click the Ad"
            color = "green"
        else:
            message = "❌ Recommended Ad Type: User Unlikely to Click the Ad"
            color = "red"

        return render_template("index.html", prediction_text=message, text_color=color)
    except Exception as e:
        return render_template("index.html", prediction_text=f"⚠️ Error: {str(e)}", text_color="orange")

if __name__ == "__main__":
    app.run(debug=True)
