# HealthPredict
Senior Project for Computer Science

Dependencies for keras and flask
(Run the following pip install commands to get these)

`>pip install numpy`

`>pip install pandas`

`>pip install tensorflow`

`>pip install keras`

`>pip install flask`

Running flask apps 

To run a flask web app, navigate to the flask_apps directory and set the environment variable flask apps. You can do this in windows with the following command
`set FLASK_APP=Health_Predict.py` 

On MacOS, the commands are: 
`export FLASK_APP=Health_Predict.py` 

You can then run the app with the following command. 
`flask run --host=0.0.0.0` 

You can then reach the app by going to http:/localhost:5000/home.
You can also replace localhost with your computer's local ip address. This will probably changed for our actual deployment, but for development purposes it will probably work for now.

