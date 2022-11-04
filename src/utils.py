import keras
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Model():
    class_name = os.path.basename(__file__)
    def __init__(self) -> None:
        #load model and scaler
        self.model=keras.models.load_model("model/model.h5")
        print("_________________________________________________________________")
        print(f"\nLog : Loading Model\n")
        print(self.model.summary())
        print("_________________________________________________________________")
        print(f"\nLog : Loading Scaler\n")
        self.scaler = MinMaxScaler(feature_range=(0,1))


    def predict(self,file):
        #Load dataset and remove dataset id column
        df=pd.read_csv(file.file)
        df=df.drop(columns="datasetId")
        #to nd array
        test_data=df.to_numpy()
        #min-max normalization
        normalized_data = self.scaler.fit_transform(test_data)
        #prediction
        print("_________________________________________________________________")
        print(f"\nLog : Predicting...\n")
        predictions=self.model.predict(normalized_data)
        #make dir to store predictions
        predictions="predictions/"
        os.makedirs(predictions,exist_ok=True)
        #convert to pandas dataframe
        df_predictions = pd.DataFrame(predictions)
        #save
        df_predictions.to_csv("predictions/predictions.csv")