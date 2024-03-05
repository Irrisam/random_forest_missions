import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
    
class Model(nn.Module):
    def __init__(self, in_features=13, h1=128, h2=128, h3=128, h4=128, out_features=2):  # Change out_features to the number of classes
        super(Model, self).__init__()

        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, out_features)
        
        self.data = pd.read_csv("/Users/tristan/Desktop/Analyse_DB/random_forest_missions/datas_training_2023.csv")
        #self.data = self.data.drop("matching_count",axis=1)
        self.data["matching_count"] = self.data["matching_count"].fillna(0)
        self.data = self.data.drop(["cancelled_count"], axis=1)
        self.X, self.y, self.label_encoders = self.data_formating()
        self.categorical_columns = ["service_id", "establishment_type", "city_id", "day_night", "mission_anticipation"]
        
        self.X = self.X.values 
        self.y = self.y.values    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)
        
        self.y_train = torch.LongTensor(self.y_train)
        self.y_test = torch.LongTensor(self.y_test)
        self.epochs = 400
        self.losses = []
        self.correct = 0
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.start = timer()


        #epoch: 84520  loss: 0.454618901014328 lr = 0.0001
        #corrects: 6114 sur un total de 12219 lr = 0.0001

    def data_formating(self):
        self.data['contractualized_count'] = self.data['contractualized_count'].apply(lambda x: 1 if x > 1 else x)
        # x = features y = cible
        self.data = self.data.drop("announcement_id", axis = 1)
        X = self.data.drop("contractualized_count", axis=1)
        y = self.data["contractualized_count"]
            
        label_encoders = {}
        categorical_columns = ["service_id", "establishment_type",  "city_id", "day_night","mission_duration", "mission_weekday","mission_anticipation"]
        # transformation des données catégorielles via vectorisation
        for column in categorical_columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
        
        for column, le in le.items():
            print(f"{column} Encoder Classes:")
            print(le.classes_)

        #for column, le in label_encoders.items():
        #        print(f"{column} Encoder Dictionary:")
        #        print(dict(zip(le.classes_, le.transform(le.classes_))))
        return X, y, label_encoders

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x
    
    #epoch: 14990  loss: 0.497355192899704
    #corrects: 9099 sur un total de 12219
    
    #epoch: 14990  loss: 0.4307946562767029
    #corrects: 9651 sur un total de 12219
    #h1= 16, h2=16
    
    #epoch: 14990  loss: 0.4291157126426697
    #corrects: 9563 sur un total de 12219
    #h1= 16, h2=16, h3=16
    
    #epoch: 14990  loss: 0.39324215054512024
    #corrects: 9634 sur un total de 12219
    # h1=32, h2=32, h3=32
    
    #epoch: 14990  loss: 0.425212025642395
    #corrects: 9642 sur un total de 12219
    #266.2112067499984
    # h1=32, h2=32, h3=32, h4=32
    
    #epoch: 14990  loss: 0.38792890310287476
    #corrects: 9823 sur un total de 12219
    #731.5601280419942
    #h1=64, h2=64, h3=64, h4=64
    
    
    # h1=128, h2=128, h3=128, h4=128
    
    
    #def forward(self, x):
    #    x = torch.sigmoid(self.fc1(x))
    #    x = torch.sigmoid(self.fc2(x))
    #    x = self.out(x)
    #    return x
    # epoch: 14990  loss: 0.5006474256515503
    # corrects: 9194 sur un total de 12219
    
    #def forward(self, x):
    #    x = torch.tanh(self.fc1(x))
    #    x = torch.tanh(self.fc2(x))
    #    x = self.out(x)
    #    return x
    #epoch: 14990  loss: 0.4828268587589264
    #corrects: 9357 sur un total de 12219
    
    #def swish(self, x):
    #    return x * torch.sigmoid(x)
    #
    #def forward(self, x):
    #    x = self.swish(self.fc1(x))
    #    x = self.swish(self.fc2(x))
    #    x = self.swish(self.fc3(x))
    #    x = self.out(x)
    #    return x
    
    #epoch: 14990  loss: 0.4764400124549866
    #corrects: 9304 sur un total de 12219
    # h1 = 8, h2 =8
    
    #epoch: 14990  loss: 0.41349226236343384
    #corrects: 9591 sur un total de 12219
    # h1=32, h2=32, h3 =32, h4 =32
    # 425.2696499169979

        

    def train(self):
        for i in range(self.epochs):
            y_pred = self.forward(self.X_train)
            loss = self.criterion(y_pred, self.y_train)
            #print(type(loss))
            self.losses.append(loss.detach().numpy())
            if i % 10 == 0:
                print(f'epoch: {i}  loss: {loss}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        plt.plot(range(self.epochs), self.losses)
        plt.ylabel("loss")
        plt.xlabel('epoch')
        #plt.show()
        total = 0
        with torch.no_grad():
            #self.y_eval = model.forward(self.X_test)
            #loss = self.criterion(self.y_eval, self.y_test)
            for i, data in  enumerate(self.X_test):
                y_val = self.forward(data)
                #print(f'{i+1}.)  {str(y_val)} \t {self.y_test[i]}')
                total += 1
                if y_val.argmax().item() == self.y_test[i]:
                    self.correct += 1
            print(f'corrects: {self.correct} sur un total de {total}')
            
    def predict_single_mission(self, mission_features):
        mission_features = pd.DataFrame(mission_features, index=[0])
        mission_features = mission_features.drop(["announcement_id", "matching_count"], axis=1)

        for column, le in self.label_encoders.items():
            mission_features[column] = le.transform(mission_features[column])

        mission_features_tensor = torch.FloatTensor(mission_features.values)
        with torch.no_grad():
            model_output = self.forward(mission_features_tensor)
            predicted_class = torch.argmax(model_output).item()
            class_probabilities = F.softmax(model_output, dim=1).numpy()
        print(predicted_class, class_probabilities[0])
        return predicted_class, class_probabilities[0]

#stat_model_instance = StatModels(None, True, False, True, "contractualized_count")
#stat_model_instance.random_forest_grid_search()

model = Model()
model.train()
mission = pd.read_csv("/Users/tristan/Desktop/Analyse_DB/random_forest_missions/single_query.csv")
mission_row = mission.iloc[0].to_dict()
model.predict_single_mission(mission_row)
end = timer()
print(end - model.start)
