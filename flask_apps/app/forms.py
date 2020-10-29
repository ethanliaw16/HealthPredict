from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, DecimalField, RadioField, IntegerField, SelectField, widgets, SelectMultipleField
from wtforms.validators import DataRequired, Optional
from wtforms.widgets import Input

# class MultiCheckboxField(SelectMultipleField):
#     widget = widgets.ListWidget(prefix_label=False)
#     option_widget = widgets.CheckboxInput()

class DiabetesInputsForm (FlaskForm):
    age = IntegerField('Age', validators=[Optional()])
    gender = SelectField('Gender', choices=[('0', 'Female'), ('1', 'Male')], validators=[DataRequired()])
    height = DecimalField('Height', places=2, validators=[DataRequired()])
    weight = DecimalField('Weight', places=2, validators=[DataRequired()])
    sys_bp = IntegerField('Systolic Blood Pressure', validators=[DataRequired()])
    dias_bp = IntegerField('Diastolic Blood Pressure', validators=[DataRequired()])
    
    smoking_history = RadioField('Smoking History', choices=[('Has never smoked', 'Has never smoked'), ('Has smoked in the past', 'Has smoked in the past')], validators=[DataRequired()])
    smoking_status = RadioField('Smoking Status', choices=[('Does not smoke', 'Does not smoke'), ('Currently smokes <1 pack daily', 'Currently smokes <1 pack daily'), ('Currently smokes 1-2 packs daily', 'Currently smokes 1-2 packs daily'), ('Currently smokes >2 packs daily', 'Currently smokes >2 packs daily')], validators=[DataRequired()])
    
    hypertension = SelectField('Hypertension', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    high_chol = SelectField('High Cholestrol', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    renal_failure = SelectField('Chronic Renal Failure', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')])
    alc_dependence = SelectField('Alcohol Dependence', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')])
    hyperchol = SelectField('Hypercholesterolemia', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')])
    coronary_artery = SelectField('Coronary Artery Disease', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')])    
    
    submit = SubmitField('Submit')

class HeartDiseaseInputsForm (FlaskForm):
    age = IntegerField('Age', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[(0, 'Female'), (1, 'Male')], validators=[DataRequired()])
    chest_pain = SelectField('Chest Pain', choices=[(0, 'Asymptomatic'), (1, 'Atypical Angina'), (2, 'Non-anginal Pain'), (3, 'Typical Angina')], validators=[DataRequired()])
    bp = IntegerField('Blood Pressure', validators=[DataRequired()])
    chol = IntegerField('Serum Cholestrol', validators=[DataRequired()])
    blood_sugar = SelectField('Fasting Blood Sugar', choices=[(0, 'False'), (1, 'True')], validators=[DataRequired()])
    ekg = SelectField('Resting Electrocardiographic Results', choices=[(0, 'Left Ventricular Hypertrophy'), (1, 'Normal'), (2, 'Having ST-T wave abnormality')], validators=[DataRequired()])
    heart_rate = IntegerField('Maximum Heart Rate Achieved During Exercise', validators=[DataRequired()])
    exercise_angina = SelectField('Exercised-induced Angina', choices=[(1, 'Yes'), (0, 'No')], validators=[DataRequired()])
    st_depression = SelectField('ST Depression Induced by Exercise Relative to Rest',  choices=[(1, 'Yes'), (0, 'No')], validators=[DataRequired()])
    slope = SelectField('Slope of the Peak Exercise ST Segment', choices=[(0, 'Downsloping'), (1, 'Flat'), (2, 'Upsloping')], validators=[DataRequired()])

    submit = SubmitField('Submit')
