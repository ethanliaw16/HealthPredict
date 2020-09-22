from flask import Flask, jsonify, request, render_template
from app import app
import pandas as pd
from app.forms import DiabetesInputsForm

# @app.route('/home', methods=['GET'])
# def get_data():
#     some_data = pd.read_csv('../ehr_diabetes_no_missing_3k.csv')
#     response = {'diabetes_data':some_data.head().to_json()}
#     return jsonify(response)

@app.route('/')
@app.route('/home', methods=['GET'])
def home():
    return render_template('homepage.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/diabetesinputs", methods=['GET', 'POST'])
def inputDiabetesInfo():
    form = DiabetesInputsForm()
    # if form.validate_on_submit():
        # send to GAN

        # return redirect(url_for('home')) this should go to output page instead
    return render_template('input_diabetes.html', title='Diabetes Inputs', form=form)
