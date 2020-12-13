from flask import Flask, jsonify, request, render_template
from app import app
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
from app.forms import DiabetesInputsForm, HeartDiseaseInputsForm
from app.diabetes_form_map import map_form_to_vector
from app.heart_disease_form_map import map_form_to_vec
from app.scale_prediction_output import scale_output, scale_input

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
        gbm_scalers = np.loadtxt('../data/gbm_scalers.csv')
        scaled_outcome = scale_output(outcome,gbm_scalers)
        output_value = round(100 * scaled_outcome[0], 3)
        return render_template('output_diabetes.html', prediction=output_value) # go to output page

    return render_template('input_diabetes.html', title='Diabetes Inputs', form=form) # go back to form

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
    if form.validate_on_submit():
        user_input = map_form_to_vec(form)
        pkl_filename = './app/HeartDisease/heart_disease_model.pkl'
        heartDiseaseLoadedModel = pickle.load(open(pkl_filename, 'rb'))
        
        input_as_arr = np.asarray(user_input)
        
        loaded_means = np.loadtxt('./app/HeartDisease/heart_disease_means.csv')
        loaded_stds = np.loadtxt('./app/HeartDisease/heart_disease_stds.csv')
        
        scaled_input = scale_input(input_as_arr, loaded_stds, loaded_means)
        
        print('Vector mapped and scaled from user input: ', scaled_input)
        predictionProbability = heartDiseaseLoadedModel.predict_proba(scaled_input)
        prediction = heartDiseaseLoadedModel.predict(scaled_input)
        outcome = round(100 * predictionProbability[0][0], 3)
        print('Our prediction: ', outcome)
        return render_template('output_heart_disease.html', prediction=outcome)
    return render_template('input_heart_disease.html', title='Heart Disease Inputs', form=form) # go back to form

@app.route('/heartdiseaseoutput')
def output_heart_disease():
    form = HeartDiseaseInputsForm()
    user_input = map_form_to_vec(form)
    outcome = {0}#= hdi.testModelWithInputs(user_input)
    print('Chance of Heart Disease: ', outcome[0])
    return render_template('output_heart_disease.html')
