#!/usr/bin/env python
# coding: utf-8

# In[1]:

def support_vertor_machine(dataset_location):
    import pandas as pd
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split 
    from sklearn.metrics import confusion_matrix, accuracy_score


    # In[2]:


    dataset = pd.read_csv(dataset_location, encoding = 'latin-1')
    # dataset.head()           #show first 5 rows


    # In[3]:


    #Combinig attributes into single list of tuples and using those features create a 2D matrix 

    # features = ['name_wt','statuses_count', 'followers_count', 'friends_count','favourites_count','listed_count']
    data = dataset.values


    # In[ ]:





    # In[ ]:


    # print("Total instances : ", data.shape[0], "\nNumber of features : ", data.shape[1])


    # In[4]:


    #convert label column into 1D arrray

    label = np.array(dataset['label'])
    # label


    # ## Test and Train Split
    # 
    # Using 80-20 split

    # In[5]:


    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)


    # In[ ]:


    # print("Number of training instances: ", X_train.shape[0])


    # In[ ]:


    # print("Number of testing instances: ", X_test.shape[0])


    # ## Training the Model

    # In[ ]:


    # Generate the model
    svm_model = SVC(kernel='linear')

    # Train the model using the training sets
    data = X_train
    label = y_train

    svm_model.fit(data, label)


    # ## Testing the Model
    # 
    # Now our model is ready. We will test our data against given labels.

    # In[ ]:


    #test set
    # X_test


    # In[ ]:


    svm_model.predict([X_test[1]])    #testing for single instance


    # In[ ]:


    '''
    Now, apply the model to the entire test set and predict the label for each test example

    '''       
        
    y_predict = []                       #to store prediction of each test example

    for test_case in range(len(X_test)): 
        label = svm_model.predict([X_test[test_case]])
        
        #append to the predictions list
        y_predict.append(np.asscalar(label))

    #predictions


    # In[ ]:


    # y_predict


    # ## Perormance evaluation of the Model

    # In[ ]:


    #true negatives is C(0,0), false negatives is C(1,0), false positives is C(0,1) and true positives is C(1,1) 
    conf_matrix = confusion_matrix(y_test, y_predict)


    # In[ ]:


    #true_negative
    TN = conf_matrix[0][0]
    #false_negative
    FN = conf_matrix[1][0]
    #false_positive
    FP = conf_matrix[0][1]
    #true_positive
    TP = conf_matrix[1][1]


    # In[ ]:


    # Recall is the ratio of the total number of correctly classified positive examples divided by the total number of positive examples. 
    # High Recall indicates the class is correctly recognized (small number of FN)
    recall = (TP)/(TP + FN)


    # In[ ]:


    # Precision is the the total number of correctly classified positive examples divided by the total number of predicted positive examples. 
    # High Precision indicates an example labeled as positive is indeed positive (small number of FP)
    precision = (TP)/(TP + FP)


    # In[ ]:


    fmeasure = (2*recall*precision)/(recall+precision)
    accuracy = (TP + TN)/(TN + FN + FP + TP)
    #accuracy_score(y_test, y_predict)


    # In[ ]:


    print("------ CLASSIFICATION PERFORMANCE OF THE SVM MODEL ------ "      "\n Recall : ", (recall*100) ,"%"       "\n Precision : ", (precision*100) ,"%"       "\n Accuracy : ", (accuracy*100) ,"%"       "\n F-measure : ", (fmeasure*100) ,"%" )

if __name__ == "__main__":
    support_vertor_machine('data/twitter_dataset.csv')