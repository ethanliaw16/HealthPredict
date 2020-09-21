from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = '8d5eb5a677571260474b96e59afd5'

from app import routes
