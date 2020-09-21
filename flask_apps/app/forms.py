from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, DecimalField
from wtforms.validators import DataRequired

class DiabetesInputsForm (FlaskForm):
    height = DecimalField('Height', places=2, validators=[DataRequired()])
    weight = DecimalField('Weight', places=2, validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    glucose = DecimalField('Glucose Level', places=2, validators=[DataRequired()])
    submit = SubmitField('Submit')

