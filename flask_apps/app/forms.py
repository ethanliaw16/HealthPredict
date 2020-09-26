from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, DecimalField, RadioField, IntegerField, SelectField, widgets, SelectMultipleField
from wtforms.validators import DataRequired

class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()

class DiabetesInputsForm (FlaskForm):
    age = IntegerField('Age')
    gender = SelectField('Gender', choices=[('Female', 'Female'), ('Male', 'Male')], validators=[DataRequired()])
    height = DecimalField('Height', places=2, validators=[DataRequired()])
    weight = DecimalField('Weight', places=2, validators=[DataRequired()])
    sys_bp = IntegerField('Systolic Blood Pressure', validators=[DataRequired()])
    dias_bp = IntegerField('Diastolic Blood Pressure', validators=[DataRequired()])
    
    smoking_status = RadioField('Smoking Status', choices=[('Has never smoked', 'Has never smoked'), ('Has smoked in the past', 'Has smoked in the past'), ('Currently smokes <1 pack daily','Currently smokes <1 pack daily'), ('Currently smokes 1-2 packs daily', 'Currently smokes 1-2 packs daily'), ('Currently smokes >2 packs daily', 'Currently smokes >2 packs daily')])
    diagnoses = MultiCheckboxField('Diagnosed with:', choices=[('Hypertension', 'Hypertension'), ('Mixed Hyperlipidemia (High Cholestrol)','Mixed Hyperlipidemia (High Cholestrol)'), ('Chronic Renal Failure', 'Chronic Renal Failure'), ('Alcohol Dependence', 'Alcohol Dependence'), ('Hypercholesterolemia', 'Hypercholesterolemia'), ('Coronary Atherosclerosis (Coronary Artery Disease)', 'Coronary Atherosclerosis (Coronary Artery Disease)')])
    
    submit = SubmitField('Submit')

