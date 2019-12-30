
def support_vector_machine(X_train, X_test, y_train, y_test):
    import pandas as pd
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split 
    from sklearn.metrics import confusion_matrix, accuracy_score

    # Generate the model
    svm_model = SVC(kernel='linear')

    # Train the model using the training sets
    data = X_train
    label = y_train

    svm_model.fit(data, label)

    y_predict = []                       #to store prediction of each test example

    for test_case in range(len(X_test)): 
        label = svm_model.predict([X_test[test_case]])
    
        #append to the predictions list
        y_predict.append(np.asscalar(label))

    return svm_model,y_predict    


