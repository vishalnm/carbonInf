from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import joblib
import folium
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler



app = Flask(__name__, template_folder="template")
model = joblib.load(open("./models/DecisionTreeClassifier.pkl", "rb"))
scaler = joblib.load(open("./models/scaler.joblib", "rb"))
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
		
		fdi_perc_gdp = float(request.form['fdi_perc_gdp'])
		
		en_per_gdp = float(request.form['en_per_gdp'])
		
		en_per_cap = float(request.form['en_per_cap'])
		
		co2_ttl = float(request.form['co2_ttl'])
		
		# co2_per_cap = float(request.form['co2_per_cap'])
		
		co2_per_gdp = float(request.form['co2_per_gdp'])
		
		pop_urb_aggl_perc= float(request.form['pop_urb_aggl_perc'])
		
		prot_area_perc = float(request.form['prot_area_perc'])
		
		#gdp = float(request.form['gdp'])
		
		gni_per_cap = float(request.form['gni_per_cap'])
		
		under_5_mort_rate = float(request.form['under_5_mort_rate'])
		
		pop_growth_perc = float(request.form['pop_growth_perc'])
		
		#pop = float(request.form['pop'])
		
		#urb_pop_growth_perc = float(request.form['urb_pop_growth_perc'])
		
		#urb_pop = float(request.form['urb_pop'])





		input_lst = [[cereal_yield,
            fdi_perc_gdp,
            en_per_gdp,
            en_per_cap,
            co2_ttl,
            
            co2_per_gdp,
            pop_urb_aggl_perc,
            prot_area_perc,
            
            gni_per_cap,
            under_5_mort_rate,
            pop_growth_perc,
           
            ]]
  
		# scaler = StandardScaler().fit(np.array(input_lst).reshape(1,-1))
		# input_data_norm = scaler.transform(np.array(input_lst).reshape(1,-1))
		
		inp_data_norm=scaler.transform(input_lst)
		
  
  		
		pred = model.predict(inp_data_norm)
		output = pred
  		
		if output == 'green':
			return render_template("green.html")
		else:
			return render_template("red.html")
	return render_template("predictor.html")



if __name__=='__main__':
	app.run(debug=True)
