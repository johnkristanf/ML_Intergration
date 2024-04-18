from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route("/result", methods=['POST'])
def predict():
    data = request.json
    experience = data.get('experience')  
    print("experience", experience)

    user_input = np.array([[float(experience)]])
    predicted_salary = model.predict(user_input)[0]
    formatted_salary = "{:,.0f}".format(predicted_salary)  

    print("formatted_salary", formatted_salary)
    return formatted_salary

@app.route("/", methods=['GET'])
def index():
    return render_template("predict_salary.html")

if __name__ == '__main__':
    app.run(debug=True)
