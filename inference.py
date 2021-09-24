import numpy as np
import pandas as pd
import pickle


pickle_in = open("Random_forest_regressor.pkl","rb")
random_forest_regressor=pickle.load(pickle_in)
url = 'https://raw.githubusercontent.com/aasim07/Complete_Data_Science_Life_Cycle_Projects/master/Data_collection/Html_scraping_data/Real_combine.csv'
df1 = pd.read_csv(url,index_col=0,parse_dates=[0])

class_names=['27.9','31.5','20','1011.5','8.5']
def predict(df):

  df = df1.iloc[:,[0,1,2,3,6]]
  
  predictions=random_forest_regressor.predict(df)
  return predictions

  
  ##return output

def main():
  predict(class_names)
  
if __name__=='__main__':
    main()
