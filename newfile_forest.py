import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def stat_models(mission_values=None, model_stats_check=False):
    
    def build_confusion_matrix(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def evaluate_mission(model, label_encoders, mission_values=None):
        """
        Method designed to assess a mission by the model:
        Input: 
                model = trained model 
                label_encoders = dict of the encoding equivalents
                mission_values (optional) = ordered list of required key/arg values (prebuilt mission for schema)
        Output:
            - predicted_count: Predicted contractualize status for the mission
            - probability_accepted_48h: Probability of the mission being contractualized within 48 hours
        """
        # assign abstract values to the mission to be tested and check the input
        # of the method to replace it if there is one
        true_value = None
        if mission_values:
            mission_df = pd.DataFrame([mission_values])
            mission_df = mission_df.drop(["announcement_id"], axis=1)

        else:
            mission_df = data.sample(n=1)
            true_value = mission_df["contractualized_count"].values[0]
            mission_df = mission_df.drop(["contractualized_count", "announcement_id"], axis=1)


        # encode with the same dictionary as the model
        for column, le in label_encoders.items():
            le.fit(X[column])  # fit on the original training data 
            mission_df[column] = mission_df[column].apply(lambda x: x if np.isin(x, le.classes_) else le.classes_[0])
            # PY 3.11 / NP 1.24.3 brittle code, update to be expected : 
            # https://stackoverflow.com/questions/46288517/getting-valueerror-y-contains-new-labels-when-using-scikit-learns-labelencoder
            # Stand off beetween NP and PY devs about comparison return types
            mission_df[column] = le.transform(mission_df[column])

        predicted_count = model.predict(mission_df)[0]
        
        predicted_proba = model.predict_proba(mission_df)[0]
        
        mission_score = predicted_proba[1] * 100   
        # prints out features and their scores for debug
        #for column in X.columns:
        #    feature_value = mission_values.get(column, 0)
        #    print(f"{column}: {feature_value}")
        if mission_values:
            print(f"mission score: {mission_score:.2f} and predicted count: {predicted_count}")    

        else:
            print(f"mission score: {mission_score:.2f} and predicted count: {predicted_count} when true value: {true_value}")    


    def gradient_booster():
        
        gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, max_depth=5, random_state=1)

        gb_model.fit(X_train, y_train)

        # Prédiction sur l'ensemble de test
        y_pred_gb = gb_model.predict(X_test)
        
        return gb_model, y_pred_gb

    def random_forest():
        
    # init d'instance pour la forest avec hyperparams opti par gridsearchCV 
        rf_model = RandomForestClassifier(max_depth=4, min_samples_split=4, min_samples_leaf=100, n_estimators=5, random_state=2)

        # training
        rf_model.fit(X_train, y_train)

        # prediction de la variable cible
        y_pred = rf_model.predict(X_test)
        
        return rf_model, y_pred

    def data_formating(data):
        # transforme les valeurs supérieurs à 1 en 1 (True)
        data['contractualized_count'] = data['contractualized_count'].apply(lambda x: 1 if x > 1 else x)
        # x = features y = cible
        data = data.drop("announcement_id", axis = 1)
        X = data.drop("contractualized_count", axis=1)
        y = data["contractualized_count"]

        # encoding des datas categorielles
        label_encoders = {}
        categorical_columns = ["specialty_id", "service_id", "establishment_type",  "city_id","department_id","region_id", "day_night","mission_duration", "mission_weekday", "mission_anticipation"]
        # transformation des données catégorielles via vectorisation
        for column in categorical_columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le

        #for column, le in label_encoders.items():
        #        print(f"{column} Encoder Dictionary:")
        #        print(dict(zip(le.classes_, le.transform(le.classes_))))
        return X, y, label_encoders
    
    def model_stats():
        # affichage des scores de précision
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Précision : {accuracy}")

        cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        print("mean values for cross validation check:", cross_val_scores.mean())
        print("standard type diff values for cross validation check:", cross_val_scores.std())

        # names y pred classes according to experience
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=1)
        print("Rapport de classification :\n", report)

        # triggers confusion matrix
        build_confusion_matrix(y_test, y_pred, labels)
    
    data = pd.read_csv("random_forest_missions/forest_updated_2023.csv")

    X, y, label_encoders = data_formating(data)

    class_names = ["Not_contractualized", "Contractualized"]
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model, y_pred = random_forest()
    evaluate_mission(model, label_encoders, mission_values)
    #print(f"mission score: {mission_score:.2f} and predicted count: {predicted_count} when true value: {true_value}")

        
    if model_stats_check == True:
        model_stats()

mission = pd.read_csv("simple_mission_test.csv")
mission_row = mission.iloc[0].to_dict()
stat_models(mission_row)