from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import joblib
import folium
from geopy.geocoders import Nominatim


app = Flask(__name__, template_folder="template")
model = joblib.load(open("./models/clf.pkl", "rb"))
print("Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
	if request.method == "POST":
		cereal_yield = float(request.form['cereal_yield'])
		# MaxTemp
		fdi_perc_gdp = float(request.form['fdi_perc_gdp'])
		# Rainfall
		en_per_gdp = float(request.form['en_per_gdp'])
		# Evaporation
		en_per_cap = float(request.form['en_per_cap'])
		# Sunshine
		co2_ttl = float(request.form['co2_ttl'])
		# Wind Gust Speed
		co2_per_cap = float(request.form['co2_per_cap'])
		# Wind Speed 9am
		co2_per_gdp = float(request.form['co2_per_gdp'])
		# Wind Speed 3pm
		pop_urb_aggl_perc= float(request.form['pop_urb_aggl_perc'])
		# Humidity 9am
		prot_area_perc = float(request.form['prot_area_perc'])
		# Humidity 3pm
		gdp = float(request.form['gdp'])
		# Pressure 9am
		gni_per_cap = float(request.form['gni_per_cap'])
		# Pressure 3pm
		under_5_mort_rate = float(request.form['under_5_mort_rate'])
		# Temperature 9am
		pop_growth_perc = float(request.form['pop_growth_perc'])
		# Temperature 3pm
		pop = float(request.form['pop'])
		# Cloud 9am
		urb_pop_growth_perc = float(request.form['urb_pop_growth_perc'])
		# Cloud 3pm
		urb_pop = float(request.form['urb_pop'])





		input_lst = [[cereal_yield,
            fdi_perc_gdp,
            en_per_gdp,
            en_per_cap,
            co2_ttl,
            co2_per_cap,
            co2_per_gdp,
            pop_urb_aggl_perc,
            prot_area_perc,
            gdp,
            gni_per_cap,
            under_5_mort_rate,
            pop_growth_perc,
            pop,
            urb_pop_growth_perc,urb_pop
            ]]
		pred = model.predict(input_lst)
		output = pred
		if output == 'green':
			return render_template("green.html")
		else:
			return render_template("red.html")
	return render_template("predictor.html")



if __name__=='__main__':
	app.run(debug=True)
