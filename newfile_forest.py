import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
    if mission_values:
        mission_df = pd.DataFrame([mission_values])
    else:
        mission_values = {
            "specialty_id": "18",
            "service_id": "32",
            "establishment_type": "clinic",
            "city_id": "35914",
            "department_id": "93",
            "region_id": "6",
            "proposed_count": "1",
            "deleted_count": "0",
            "expired_count": "0",
            "accepted_count": "1",
            "waiting_count": "1",
            "cancelled_before_contract_count": "0",
            "within_two_days": "0",
            "cancelled_with_contract_count": "0",
            "day_night": "NIGHT",
            "nbr_of_needs": "1",
            "created_day_of_week": "1",
            "mission_duration": "645"
        }
        mission_df = pd.DataFrame([mission_values])

    # encode with the same dictionary as the model
    for column, le in label_encoders.items():
        le.fit(X[column])  # fit on the original training data 
        mission_df[column] = mission_df[column].apply(lambda x: x if np.isin(x, le.classes_) else le.classes_[0])
        # PY 3.11 / NP 1.24.3 brittle code, update to be expected : 
        # https://stackoverflow.com/questions/46288517/getting-valueerror-y-contains-new-labels-when-using-scikit-learns-labelencoder
        # Stand off beetween NP and PY devs about comparison return types
        mission_df[column] = le.transform(mission_df[column])

    predicted_count = model.predict(mission_df)[0]

    probability_accepted_week = model.predict_proba(mission_df)[0][1]

    feature_importances = model.feature_importances_

    total_feature_score = 0

    # calculate the score of the mission via its features
    for column, importance in zip(X.columns, feature_importances):
        mission_value = float(mission_df[column].values[0])
        total_feature_score += mission_value * importance

    # calculate the maximum theoretical score of the features
    max_feature_score = 0
    for column, importance in zip(X.columns, feature_importances):
        max_feature_score += float(X[column].max()) * importance

    # rescale the total feature score to be within 0 to 100
    rescaled_score = (total_feature_score / max_feature_score) * 100
    
    # prints out features and their scores for debug
    #for column in X.columns:
    #    feature_value = mission_values.get(column, 0)
    #    print(f"{column}: {feature_value}")

    print(f"score total de la mission {rescaled_score:.2f} sur 100")
    print(f"Prediction de la contractualisation de la mission: {predicted_count}")
    print(f"Chances de la mission d'être contractualisée en moins d'une semaine: {probability_accepted_week}")

    return predicted_count, probability_accepted_week


data = pd.read_csv("query_notfull.csv")
# transforme les valeurs supérieurs à 1 en 1 (True)
data['contractualized_count'] = data['contractualized_count'].apply(lambda x: 1 if x > 1 else x)

# x = features y = cible
data = data.drop("announcement_id", axis = 1)
data = data.drop("within_one_day", axis = 1)
data = data.drop("within_week_count", axis = 1)
data = data.drop("salary", axis = 1)

X = data.drop("contractualized_count", axis=1)
y = data["contractualized_count"]

# encoding des datas categorielles
label_encoders = {}
categorical_columns = ["specialty_id", "service_id", "establishment_type",  "city_id","department_id","region_id", "day_night", "created_day_of_week","mission_duration"]
# transformation des données catégorielles via vectorisation
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le


#for column, le in label_encoders.items():
#        print(f"{column} Encoder Dictionary:")
#        print(dict(zip(le.classes_, le.transform(le.classes_))))
# split data en trainset et testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# init de class pour la forest avec hyperparams opti par gridsearchCV 
class_weights = {0: 1, 1: 1}
model = RandomForestClassifier(max_depth=20, min_samples_split=2, n_estimators=100, random_state=42, class_weight=class_weights)

# training
model.fit(X_train, y_train)

# prediction de la variable cible
y_pred = model.predict(X_test)

# affichage des scores de précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision : {accuracy}")

# names y pred classes according to experience
class_names = ["Not_contractualized", "Contractualized"]
report = classification_report(y_test, y_pred, target_names=class_names, zero_division=1)
print("Rapport de classification :\n", report)

# triggers confusion matrix
#labels = sorted(y.unique())
#build_confusion_matrix(y_test, y_pred, labels)

predicted_count, probability_accepted_week = evaluate_mission(model, label_encoders, mission_values=None)
