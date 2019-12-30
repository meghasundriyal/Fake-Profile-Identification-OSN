
def k_nearest_neighbours(X_train, X_test, y_train, y_test):
    # print('neighbours: ',neighbors)
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split 
    from sklearn.metrics import confusion_matrix, accuracy_score

    neighbors = 5
    knn_model = KNeighborsClassifier(n_neighbors = neighbors)

    # Train the model using the training sets
    data = X_train
    label = y_train

    knn_model.fit(data, label)

    y_predict = []                       #to store prediction of each test example

    for test_case in range(len(X_test)): 
        label = knn_model.predict([X_test[test_case]])
        
        #append to the predictions list
        y_predict.append(np.asscalar(label))


    return knn_model,y_predict
