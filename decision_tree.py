def decision_tree(X_train, X_test, y_train, y_test):
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix, accuracy_score



    clf = DecisionTreeClassifier(min_impurity_decrease=0.001)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_predict = clf.predict(X_test)

    return clf,y_predict