from app.forms import DiabetesInputsForm
from app.impute_missing_for_input import impute_missing
import numpy as np

def map_form_to_vector(form):
    prediction_vector = np.zeros(20)
    prediction_vector[0] = 2020 - form.age.data
    prediction_vector[1] = form.gender.data
    prediction_vector[2] = form.height.data
    prediction_vector[3] = form.weight.data
    english_bmi = 703 * form.weight.data / (form.height.data**2)
    prediction_vector[4] = english_bmi
    prediction_vector[5] = form.sys_bp.data
    prediction_vector[6] = form.dias_bp.data
    prediction_vector[7] = form.hypertension.data
    prediction_vector[8] = form.high_chol.data
    prediction_vector[9] = form.renal_failure.data
    prediction_vector[10] = form.alc_dependence.data
    prediction_vector[11] = form.hyperchol.data
    prediction_vector[12] = form.coronary_artery.data
    smokingHistoryIndex = 16
    currentSmokeStatusIndex = -1
    if(form.smoking_history.data == 'Has smoked in the past'):
        prediction_vector[smokingHistoryIndex] = 1

    if(form.smoking_status.data == 'Currently smokes <1 pack daily'):
        currentSmokeStatusIndex = 15
    elif(form.smoking_status.data == 'Currently smokes 1-2 packs daily'):
        currentSmokeStatusIndex = 17
    elif(form.smoking_status.data == 'Currently smokes >2 packs daily'):
        currentSmokeStatusIndex = 18

    if(currentSmokeStatusIndex > 0):
        prediction_vector[currentSmokeStatusIndex] = 1
    prediction_vector[19] = 10.5
    #prediction_vector.append(form.sys_bp.data)
    #prediction_vector.append(form.dias_bp.data)
    #print('gender ', form.gender.data)
    #print('height ', form.height.data)
    #print('smoking history ', form.smoking_history.data)
    #print('smoking status ', form.smoking_status.data)
    #print('Current Vector: ', prediction_vector)
    prediction_vector_no_missing = impute_missing(prediction_vector)
    return prediction_vector_no_missing
