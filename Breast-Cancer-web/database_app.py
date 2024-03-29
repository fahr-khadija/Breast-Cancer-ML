import pandas as pd

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

from flask import Flask, jsonify, render_template
from flask_cors import CORS

#################################################
# Database Setup
#################################################

engine = create_engine('postgresql://postgres:postgres@localhost:5432/Breast_cancer')

# reflect an existing database into a new model
Base = automap_base()

# reflect the tables
Base.prepare(autoload_with=engine)

# Save references to each table
Breast_Cancer_table = Base.classes.breast_cancer_table 
# Create our session (link) from Python to the DB
session = Session(engine)

#################################################
# Flask Setup
#################################################
app = Flask(__name__)
CORS(app)

#################################################
# Flask Routes
#################################################

@app.route("/home")
def homepage():
    return render_template("index.html")

@app.route("/")
def welcome():

#List all available api routes
    return (
        f"Available Routes:<br/>"
        f"All Data: /all_data<br/>"
        f"Homepage: /home<br/>"
    )

# Return the results of the precipitation analysis
@app.route("/all_data")
def all_data():

    Breast_Cancer_data = pd.read_sql_table('Breast_Cancer', engine)
    
#return json data   
    return jsonify({'projectdata': Breast_Cancer.to_dict(orient='records')})

#close session
session.close()

if __name__ == '__main__':
    app.run(debug=True)