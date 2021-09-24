import numpy as np
import pandas as pd
import pickle


pickle_in = open("Random_forest_regressor.pkl","rb")
random_forest_regressor=pickle.load(pickle_in)

Average_Temperature= input()
Maximum_Temperature = input()
Minimum_Temperature = input()
Atm_pressure_at_sea_level = input()
Average_wind_speed = input()

class_names=[ 'Average_Temperature','Maximum_Temperature','Minimum_Temperature', 'Atm_pressure_at_sea_level','Average_wind_speed']

def predict(df):

  df = [[ Average_Temperature,Maximum_Temperature,Minimum_Temperature, Atm_pressure_at_sea_level,Average_wind_speed]]
  predictions=random_forest_regressor.predict(df)

  print(predictions)
  return predictions
  
