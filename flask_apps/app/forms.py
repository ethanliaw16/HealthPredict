from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, DecimalField, RadioField, IntegerField, SelectField, widgets, SelectMultipleField
from wtforms.validators import DataRequired
from wtforms.widgets import Input

class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()

class DiabetesInputsForm (FlaskForm):
    age = IntegerField('Age')
    # what value to put: 1 or 'Female'?
    gender = SelectField('Gender', choices=[(1, 'Female'), (2, 'Male')], validators=[DataRequired()])
    height = DecimalField('Height', places=2, validators=[DataRequired()])
    weight = DecimalField('Weight', places=2, validators=[DataRequired()])
    sys_bp = IntegerField('Systolic Blood Pressure', validators=[DataRequired()])
    dias_bp = IntegerField('Diastolic Blood Pressure', validators=[DataRequired()])
    
    # 1 or 'has never smoked'?
    smoking_history = RadioField('Smoking History', choices=[('Has never smoked', 'Has never smoked'), ('Has smoked in the past', 'Has smoked in the past')])
    smoking_status = RadioField('Smoking Status', choices=[('Does not smoke', 'Does not smoke'), ('Currently smokes <1 pack daily', 'Currently smokes <1 pack daily'), ('Currently smokes 1-2 packs daily', 'Currently smokes 1-2 packs daily'), ('Currently smokes >2 packs daily', 'Currently smokes >2 packs daily')])
    
    # what for ignore? 2?
    # hypertension = RadioField('Hypertension', choices=[('2', '--Select--'),('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    hypertension = SelectField('Hypertension', choices=[('2', '--Select--'),('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    high_chol = SelectField('High Cholestrol', choices=[('2', '--Select--'),('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    renal_failure = SelectField('Chronic Renal Failure', choices=[('2', '--Select--'),('1', 'Yes'), ('0', 'No')])
    alc_dependence = SelectField('Alcohol Dependence', choices=[('2', '--Select--'),('1', 'Yes'), ('0', 'No')])
    hyperchol = SelectField('Hypercholesterolemia', choices=[('2', '--Select--'),('1', 'Yes'), ('0', 'No')])
    coronary_artery = SelectField('Coronary Artery Disease', choices=[('2', '--Select--'),('1', 'Yes'), ('0', 'No')])

    # diagnoses = MultiCheckboxField('Diagnosed with:', choices=[('Hypertension', 'Hypertension'), ('Mixed Hyperlipidemia (High Cholestrol)','Mixed Hyperlipidemia (High Cholestrol)'), ('Chronic Renal Failure', 'Chronic Renal Failure'), ('Alcohol Dependence', 'Alcohol Dependence'), ('Hypercholesterolemia', 'Hypercholesterolemia'), ('Coronary Atherosclerosis (Coronary Artery Disease)', 'Coronary Atherosclerosis (Coronary Artery Disease)')])
    
    submit = SubmitField('Submit')

