from fastapi import FastAPI
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import sys
import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = FastAPI()

class StatModels:
    def __init__(self, mission_values, model_stats_check=False, confusion_matrix_check=False, banned_rows_toggle=True, checked_value="contractualized_count"):
        
        self.mission_values = mission_values
        self.model_stats_check = model_stats_check
        self.confusion_matrix_check = confusion_matrix_check
        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__), "datas_training_2023.csv"))
        self.checked_value = checked_value
        self.banned_rows = ["region_id", "department_id", "specialty_id"]
        self.banned_rows_toggle = banned_rows_toggle
        
        if self.checked_value == "cancelled_count":
            self.class_names = ["Not_canceled", "Canceled"]
        
        self.X, self.y, self.label_encoders = self.data_formating()
        self.labels = sorted(self.y.unique())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model, self.y_pred = self.random_forest()
        
        if mission_values:
            self.evaluate_mission()
        else:
            sys.exit()
        
        if self.confusion_matrix_check:
            self.build_confusion_matrix()

        if self.model_stats_check:
            self.model_stats()
        
    def build_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.labels, yticklabels=self.labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def evaluate_mission(self):
        """
        Method designed to assess a mission by the model:
        Input: 
                model = trained model 
                label_encoders = dict of the encoding equivalents
                mission_values (optional) = ordered list of required key/arg values (prebuilt mission for schema)
        Output:
            - predicted_count: Predicted contractualize status for the mission
        """
        # assign abstract values to the mission to be tested and check the input
        # of the method to replace it if there is one
        true_value = None
        try:
            if self.mission_values:
                mission_df = pd.DataFrame([self.mission_values])
                if self.checked_value == "contractualized_count":
                    mission_df = mission_df.drop(["contractualized_count", "announcement_id"], axis=1)
                if self.checked_value == "cancelled_count":
                    mission_df = mission_df.drop(["cancelled_count", "announcement_id"], axis=1)
            else:
                mission_df = self.data.sample(n=1)
                mission_df = mission_df.drop(["contractualized_count", "announcement_id"], axis=1)
            #mission_df.drop("salary")
            # encode with the same dictionary as the model
            if self.banned_rows_toggle == True:
                mission_df = mission_df.drop(self.banned_rows, axis=1)
            

            for column, le in self.label_encoders.items():
                le.fit(self.X[column])  # fit on the original training data 
                mission_df[column] = mission_df[column].apply(lambda x: x if np.isin(x, le.classes_) else le.classes_[0])
                # PY 3.11 / NP 1.24.3 brittle code, update to be expected : 
                # https://stackoverflow.com/questions/46288517/getting-valueerror-y-contains-new-labels-when-using-scikit-learns-labelencoder
                # Stand off beetween NP and PY devs about comparison return types
                mission_df[column] = le.transform(mission_df[column])
            
        except KeyError as e:
            raise ValueError(f"{e}. Please check the keys in self.X and ensure they match the expected columns.X") from e

        predicted_count = self.model.predict(mission_df)[0]
        
        predicted_proba = self.model.predict_proba(mission_df)[0]
                
        mission_score = (predicted_proba[1] * 100)
        # prints out features and their scores for debug
        #for column in self.X.columns:
        #    feature_value = self.mission_values.get(column, 0)
        #    print(f"{column}: {feature_value}")
        if self.mission_values:
            a = f"mission score: {mission_score:.2f} and predicted count: {predicted_count}"
            print(a)
            return a
        else:
            print(f"mission score: {mission_score:.2f} and predicted count: {predicted_count} when true value: {true_value}")


    def gradient_booster(self):
        gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, max_depth=5, random_state=1)
        gb_model.fit(self.X_train, self.y_train)
        y_pred_gb = gb_model.predict(self.X_test)
        return gb_model, y_pred_gb
    def random_forest(self):
    # init d'instance pour la forest avec hyperparams opti par gridsearchCV 
        if self.checked_value == "contractualized_count":
            #rf_model = RandomForestClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=5, n_estimators=150, random_state=2)
            rf_model = RandomForestClassifier(max_depth=None, min_samples_split=10, min_samples_leaf=2, n_estimators=150, class_weight={0:1, 1:2.5}, random_state=35)
        elif self.checked_value == "cancelled_count":
            rf_model = RandomForestClassifier(max_depth=None, min_samples_split=10, min_samples_leaf=2, n_estimators=150, class_weight= "balanced", random_state=42)
            #rf_model = RandomForestClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=5, n_estimators=50, random_state=2)

        # training
        rf_model.fit(self.X_train, self.y_train, sample_weight=1)

        # prediction de la variable cible
        y_pred = rf_model.predict(self.X_test)
        
        return rf_model, y_pred

    def data_formating(self):
        self.data['contractualized_count'] = self.data['contractualized_count'].apply(lambda x: 1 if x > 1 else x)
        # x = features y = cible
        if self.banned_rows_toggle is True:
            self.data = self.data.drop(self.banned_rows, axis=1)
        self.data = self.data.drop("announcement_id", axis = 1)
        if self.checked_value == "contractualized_count":
            X = self.data.drop("contractualized_count", axis=1)
            y = self.data["contractualized_count"]
        elif self.checked_value == "cancelled_count":
            X = self.data.drop("cancelled_count", axis=1)
            y = self.data["cancelled_count"]

        # encoding des datas categorielles
        label_encoders = {}
        if self.checked_value == "contractualized_count":
            categorical_columns = ["service_id", "establishment_type",  "city_id", "day_night","mission_duration", "mission_weekday", "cancelled_count","mission_anticipation"]
        elif self.checked_value == "cancelled_count":
            categorical_columns = ["service_id", "establishment_type",  "city_id", "day_night","mission_duration", "mission_weekday", "contractualized_count","mission_anticipation"]
        # transformation des données catégorielles via vectorisation
        for column in categorical_columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le

        #for column, le in label_encoders.items():
        #        print(f"{column} Encoder Dictionary:")
        #        print(dict(zip(le.classes_, le.transform(le.classes_))))
        return X, y, label_encoders
    def model_stats(self):
        # affichage des scores de précision
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Précision : {accuracy}")

        cross_val_scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring='accuracy')

        print("mean values for cross validation check:", cross_val_scores.mean())
        print("standard type diff values for cross validation check:", cross_val_scores.std())

        # names y pred classes according to experience
        report = classification_report(self.y_test, self.y_pred, zero_division=1, target_names=["not contractualized", "contractualized"])
        print("Rapport de classification :\n", report)
    
    def unittest_link(self):
        return self.data, self.model, self.label_encoders, self.mission_values
    
    def random_forest_grid_search(self):
        # hyperparameter grid for RandomForestClassifier
        param_grid = {
            'n_estimators': [25, 50, 100, 150, 200],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': [1, 2, 5, 8, 10, 15, 18, 20],
            'min_samples_leaf': [2, 5, 7, 10, 15],
            'random_state': [42]
        }

        # create RandomForestClassifier instance
        rf_model = RandomForestClassifier()

        # instantiate GridSearchCV
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='f1')

        grid_search.fit(self.X_train, self.y_train)

        # get the best parameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        best_rf_model = RandomForestClassifier(**best_params)
        best_rf_model.fit(self.X_train, self.y_train)

      
        y_pred = best_rf_model.predict(self.X_test)

        return best_rf_model, y_pred

def api_query_call(announcement_id = None):    
    response_data = {"id": "66badd9d-66bb-4227-9781-daff02a5383a"}
    session_id = response_data['id']
    headers = {'X-Metabase-Session': session_id, 'Content-Type': 'application/json'} #session identification token for metabase API
    
    metabase_api_url = f"https://metabase.medelse.com/api/card/893/query/csv" #csv typed query
    payload = [ {"type": "number", "value": announcement_id, "target": [ "variable", ["template-tag", "announcement_id"] ] } ] #query payload
    
    with requests.Session() as session:
        session.headers.update(headers) #update headers for metabase API identification
        if announcement_id is not None:
            response = session.post(url=metabase_api_url + "?parameters=" + json.dumps(payload)) #rebuilt url to get filtered query from metabase API
        else:
            response = session.post(url=metabase_api_url) #unrequested ID
        response.raise_for_status() #raises error numbers
        
        csv_file_path = os.path.join(os.path.dirname(__file__), "single_query.csv") #finds path to mission csv within local machine
        with open(csv_file_path, "wb") as csv_file:
            csv_file.write(response.content)
        print(f"CSV result saved to {csv_file_path}")


@app.get("/{announcement_id}")
def run_with_id(announcement_id):
    int_check = all(char.isdigit() for char in announcement_id)#check for ints in inputed id (which is str)
    if int_check is False:
        return {"Please input a valid numeric announcement_id"}
    api_query_call(announcement_id)
    try:
        csv_file_path = os.path.join(os.path.dirname(__file__), "single_query.csv")
        mission = pd.read_csv(csv_file_path)
        mission_row = mission.iloc[0].to_dict()#fetches the IDed mission line
        stat_model_instance = StatModels(mission_row, False, False, True, "contractualized_count")
        result = stat_model_instance.evaluate_mission()
        return {result}
    except IndexError:
        return {"Please input a valid announcement_id"} #if ID prints nothing
    #stat_model_instance.random_forest_grid_search()

@app.get("/")
def run():
    api_query_call()
    csv_file_path = os.path.join(os.path.dirname(__file__), "FS_single_query.csv") #safe mission csv file for ungiven id
    mission = pd.read_csv(csv_file_path)
    mission_row = mission.iloc[0].to_dict()
    stat_model_instance = StatModels(mission_row, False, False, True, "contractualized_count")
    result = stat_model_instance.evaluate_mission()
    return {result}


# docker run -p 8000:8000 random_forest_docker
