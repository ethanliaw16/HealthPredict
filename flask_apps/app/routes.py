from flask import Flask, jsonify, request, render_template
from app import app
import pandas as pd
import lightgbm as lgb
import pickle
from app.forms import DiabetesInputsForm, HeartDiseaseInputsForm
from app.diabetes_form_map import map_form_to_vector
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
    if form.validate_on_submit():
        # send to GAN
        user_input = map_form_to_vector(form)
        gbm_predictor = pickle.load(open('../trained_models/gbm_predictor.txt', 'rb'))
        outcome = gbm_predictor.predict([user_input], num_iteration=gbm_predictor.best_iteration)
        print('Chance of type 2 Diabetes: ', outcome[0])
        output_value = round(100 * outcome[0], 3)
        return render_template('output_diabetes.html', prediction=output_value) #this should go to output page instead

        # return redirect(url_for('diabetesoutput'))
    return render_template('input_diabetes.html', title='Diabetes Inputs', form=form)

@app.route('/diabetesoutput')
def output_diabetes():
    default_input = [[1970, 67, 180, 28.2, 120, 80, 1, 1, 0, 0, 0, 1, 0, 0,0,0,0,0,0]]
    gbm_predictor = pickle.load(open('../trained_models/gbm_predictor.txt', 'rb'))
    outcome = gbm_predictor.predict(default_input, num_iteration=gbm_predictor.best_iteration)
    print('Chance of type 2 Diabetes: ', outcome[0])
    return render_template('output_diabetes.html')

@app.route('/heartdiseaseinputs', methods=['GET', 'POST'])
def inputHeartDiseaseInfo():
    form = HeartDiseaseInputsForm()
    # if form.validate_on_submit():
         # return redirect(url_for('home')) this should go to output page instead
    return render_template('input_heart_disease.html', title='Heart Disease Inputs', form=form)

@app.route('/heartdiseaseoutput')
def output_heart_disease():
    return render_template('output_heart_disease.html')
