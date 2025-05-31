import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Make sure you're loading the correct model (not scaler unless you meant to)
model = pickle.load(open('score.pkl', 'rb'))  

grade_mapping = {
    0: 'A:Excellent',
    1: 'B:Very Good',
    2: 'C:Good',
    3: 'D:Pass',
    4: 'F:Fail'
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    english_score = float(request.form['English'])
    maths_score = float(request.form['Maths'])
    biology_score = float(request.form['Biology'])
    chemistry_score = float(request.form['Chemistry'])
    physics_score = float(request.form['Physics'])
    geography_score = float(request.form['Geography'])
    computer_score = float(request.form['Computer'])

    final_features = np.array([[english_score, maths_score, biology_score, chemistry_score, physics_score, geography_score, computer_score]])

    prediction_num = model.predict(final_features)[0]

    # Map numeric prediction to grade label
    prediction_label = grade_mapping.get(prediction_num, "Unknown")

    return render_template(
        'index.html',pred=prediction_label, 
        english=english_score,
        maths=maths_score,
        biology=biology_score,
        chemistry=chemistry_score,
        physics=physics_score,
        geography=geography_score,  
        computer=computer_score,
    )

if __name__ == "__main__":
    app.run(debug=True)
