from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, DecimalField, RadioField, IntegerField, SelectField, widgets, SelectMultipleField
from wtforms.validators import DataRequired
from wtforms.widgets import Input

# class MultiCheckboxField(SelectMultipleField):
#     widget = widgets.ListWidget(prefix_label=False)
#     option_widget = widgets.CheckboxInput()

class DiabetesInputsForm (FlaskForm):
    age = IntegerField('Age')
    # what value to put: 1 or 'Female'?
    gender = SelectField('Gender', choices=[(0, 'Female'), (1, 'Male')], validators=[DataRequired()])
    height = DecimalField('Height', places=2, validators=[DataRequired()])
    weight = DecimalField('Weight', places=2, validators=[DataRequired()])
    sys_bp = IntegerField('Systolic Blood Pressure', validators=[DataRequired()])
    dias_bp = IntegerField('Diastolic Blood Pressure', validators=[DataRequired()])
    
    # 1 or 'has never smoked'?
    smoking_history = RadioField('Smoking History', choices=[('Has never smoked', 'Has never smoked'), ('Has smoked in the past', 'Has smoked in the past')])
    smoking_status = RadioField('Smoking Status', choices=[('Does not smoke', 'Does not smoke'), ('Currently smokes <1 pack daily', 'Currently smokes <1 pack daily'), ('Currently smokes 1-2 packs daily', 'Currently smokes 1-2 packs daily'), ('Currently smokes >2 packs daily', 'Currently smokes >2 packs daily')])
    
    # what for ignore? -1
    # hypertension = RadioField('Hypertension', choices=[('2', '--Select--'),('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    hypertension = SelectField('Hypertension', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    high_chol = SelectField('High Cholestrol', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    renal_failure = SelectField('Chronic Renal Failure', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')])
    alc_dependence = SelectField('Alcohol Dependence', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')])
    hyperchol = SelectField('Hypercholesterolemia', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')])
    coronary_artery = SelectField('Coronary Artery Disease', choices=[('-1', '--Select--'),('1', 'Yes'), ('0', 'No')])

    # diagnoses = MultiCheckboxField('Diagnosed with:', choices=[('Hypertension', 'Hypertension'), ('Mixed Hyperlipidemia (High Cholestrol)','Mixed Hyperlipidemia (High Cholestrol)'), ('Chronic Renal Failure', 'Chronic Renal Failure'), ('Alcohol Dependence', 'Alcohol Dependence'), ('Hypercholesterolemia', 'Hypercholesterolemia'), ('Coronary Atherosclerosis (Coronary Artery Disease)', 'Coronary Atherosclerosis (Coronary Artery Disease)')])
    
    submit = SubmitField('Submit')

class HeartDiseaseInputsForm (FlaskForm):
    age = IntegerField('Age', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[(1, 'Female'), (2, 'Male')], validators=[DataRequired()])
    chest_pain = SelectField('Chest Pain', choices=[(1, 'Asymptomatic'), (2, 'Atypical Angina'), (3, 'Non-anginal Pain'), (4, 'Typical Angina')], validators=[DataRequired()])
    bp = IntegerField('Blood Pressure', validators=[DataRequired()])
    chol = IntegerField('Serum Cholestrol', validators=[DataRequired()])
    blood_sugar = IntegerField('Fasting Blood Sugar', validators=[DataRequired()])
    ekg = SelectField('Resting Electrocardiographic Results', choices=[(1, 'Left Ventricular Hypertrophy'), (2, 'Normal'), (3, 'Having ST-T wave abnormality')], validators=[DataRequired()])
    heart_rate = IntegerField('Maximum Heart Rate Achieved During Exercise', validators=[DataRequired()])
    exercise_angina = SelectField('Exercised-induced Angina', choices=[(1, 'Yes'), (0, 'No')], validators=[DataRequired()])
    st_depression = SelectField('ST Depression Induced by Exercise Relative to Rest',  choices=[(1, 'Yes'), (0, 'No')], validators=[DataRequired()])
    slope = SelectField('Slope of the Peak Exercise ST Segment', choices=[(1, 'Downsloping'), (2, 'Flat'), (3, 'Upsloping')], validators=[DataRequired()])

    submit = SubmitField('Submit')
