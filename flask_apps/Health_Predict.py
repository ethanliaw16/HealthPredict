# from flask import Flask, jsonify, request, render_template
# import pandas as pd
# from forms import DiabetesInputsForm
# app = Flask(__name__)

# # @app.route('/home', methods=['GET'])
# # def get_data():
# #     some_data = pd.read_csv('../ehr_diabetes_no_missing_3k.csv')
# #     response = {'diabetes_data':some_data.head().to_json()}
# #     return jsonify(response)

# @app.route('/home', methods=['GET'])
# def home():
#     return render_template('homepage.html')

# @app.route("/create", methods=['GET', 'POST'])
# def inputDiabetesInfo():
#     form = DiabetesInputsForm()
#     # if form.validate_on_submit():
#         # send to GAN instead of Post

#         # inputs = Post(title=form.title.data, edition=form.edition.data, authors=form.authors.data,
#         # price=form.price.data, user_id=current_user.id, class_id=form.course.data,
#         # quality=form.quality.data, description=form.description.data)
    
#         # return redirect(url_for('home')) this should go to output page instead
#     return render_template('input_diabetes.html', form=form)

from app import app
