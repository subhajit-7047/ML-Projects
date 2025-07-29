from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize the app
app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load('pipeline_1.pkl')

# Home route
@app.route('/')
def home():   
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        pclass = int(request.form['Pclass'])
        sex = request.form['Sex']
        age = float(request.form['Age'])
        sibsp = int(request.form['SibSp'])
        parch = int(request.form['Parch'])
        fare = float(request.form['Fare'])
        embarked = request.form['Embarked']

        # Create DataFrame
        input_data = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked
        }])

        # Predict
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][prediction]

        result = "🎉 Survived!" if prediction == 1 else "💀 Did not survive"
        prob = f"{probability:.2f}"

        return render_template('index.html', result=result, probability=prob)

if __name__ == '__main__':
    app.run(debug=True)
