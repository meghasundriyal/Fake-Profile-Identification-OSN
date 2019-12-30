import pandas as pd
import numpy as np
import k_nearest_neighbors as my_k_nearest_neighbors
import neural_network as my_neural_network
import decision_tree as my_decision_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


dataset_location='data/twitter_dataset.csv'
dataset = pd.read_csv(dataset_location, encoding = 'latin-1')

features=[]
for attributes in dataset.columns:
    if attributes != 'label':
        features.append(attributes)

X = dataset.as_matrix(columns = features) # Features
y = dataset.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 


knn_model,knn_y_predict=my_k_nearest_neighbors.k_nearest_neighbours(X_train, X_test, y_train, y_test)
print("trained knn")
dt_model,dt_y_predict=my_decision_tree.decision_tree(X_train, X_test, y_train, y_test)
print("trained decision tree")
nn_model,nn_y_predict=my_neural_network.neural_network(X_train, X_test, y_train, y_test)
print("neural network")

combined_model_y=[]
for i in range(len(y_test)):
    if(nn_y_predict[i]+knn_y_predict[i]+dt_y_predict[i] >1):
        combined_model_y.append(1)
    else:
        combined_model_y.append(0)

conf_matrix = confusion_matrix(y_test, combined_model_y)

#true_negative
TN = conf_matrix[0][0]
#false_negative
FN = conf_matrix[1][0]
#false_positive
FP = conf_matrix[0][1]
#true_positive
TP = conf_matrix[1][1]

recall = (TP)/(TP + FN)

precision = (TP)/(TP + FP)

fmeasure = (2*recall*precision)/(recall+precision)

accuracy = (TP + TN)/(TN + FN + FP + TP)

print("------ CLASSIFICATION PERFORMANCE OF THE SVM MODEL ------ "\
      "\n Recall : ", (recall*100) ,"%" \
      "\n Precision : ", (precision*100) ,"%" \
      "\n Accuracy : ", (accuracy*100) ,"%" \
      "\n F-measure : ", (fmeasure*100) ,"%" )









