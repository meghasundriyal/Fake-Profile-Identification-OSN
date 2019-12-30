def neural_network(X_train, X_test, y_train, y_test):
    from numpy import loadtxt
    from keras.models import Sequential
    from keras.layers import Dense
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split 
    from sklearn.metrics import confusion_matrix, accuracy_score

    length_of_features=X_train.shape[1] #columns in dataframe X_train

    model = Sequential()

    model.add(Dense(15, input_dim=length_of_features, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=150, batch_size=150)
    
    y_predict =   model.predict_classes(X_test)  

    return model,y_predict
