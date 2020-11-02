from app.forms import HeartDiseaseInputsForm
import numpy as np

def map_form_to_vec(form):
    prediction_vector = np.zeros(13)
    prediction_vector[0] = form.age.data
    prediction_vector[1] = form.gender.data
    prediction_vector[2] = form.chest_pain.data
    prediction_vector[3] = form.bp.data
    prediction_vector[4] = form.chol.data
    prediction_vector[5] = form.blood_sugar.data
    prediction_vector[6] = form.ekg.data
    prediction_vector[7] = form.heart_rate.data
    prediction_vector[8] = form.exercise_angina.data
    prediction_vector[9] = form.st_depression.data
    prediction_vector[10] = form.slope.data
    prediction_vector[11] = 0
    prediction_vector[12] = 3
    return prediction_vector
