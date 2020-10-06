from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = '5aOlWgefqlYW09ZZfwJL7g2y3e5bR89x'

from app import routes
