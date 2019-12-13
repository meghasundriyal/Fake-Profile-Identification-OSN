#!/usr/bin/env python
# coding: utf-8

# In[5]:

def pca(dataset_location,new_dataset_name,number_of_components=4):
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA


    # In[6]:


    df = pd.read_csv(dataset_location, encoding = 'latin-1')


    # In[7]:


    from sklearn.preprocessing import StandardScaler
    features = ['name_wt','statuses_count', 'followers_count', 'friends_count','favourites_count','listed_count']
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,['label']].values
    x = StandardScaler().fit_transform(x)


    # In[16]:


    # number_of_components=4
    pca = PCA(n_components=number_of_components)
    principalComponents = pca.fit_transform(x)
    column_names= [] 

    for i in range(1,number_of_components+1):
        
        column_names.append('principal_component_'+str(i))
        
    principalDf = pd.DataFrame(data = principalComponents
                , columns = column_names)

    principalDf['label']=y
    # In[17]:


    # print(principalDf)
    export_csv = principalDf.to_csv (new_dataset_name, index = None, header=True) 
    return  new_dataset_name

if __name__ == "__main__":
    pca('data/twitter_dataset.csv','data/twitter_dataset_4.csv',4)
# In[ ]:




