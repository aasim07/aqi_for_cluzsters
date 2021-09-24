import numpy as np
import pandas as pd
import pickle


pickle_in = open("Random_forest_regressor.pkl","rb")
random_forest_regressor=pickle.load(pickle_in)

def predict(df):
  df=df[['T','TM','Tm','SLP','H','VV','V','VM','PM 2.5']]
  df1=df.iloc[:,[0,1,2,3,6]]
  predictions=random_forest_regressor.predict(df1)[0]
  print(predictions)
  return predictions

  
  ##return output

def main():
  combine_data= pd.read_csv(r'Real_combine.csv')
  predict(combine_data)
  
if __name__=='__main__':
    main()
