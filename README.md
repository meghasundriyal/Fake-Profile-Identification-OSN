# Fake-Profile-Identification-OSN

Here we explore various Supervised and Unsupervised techniques to check whether a twitter profile is genuine or fake. We have tested various available Machine Learning Models and have come up with a Hybrid Model , implemented in Fake_Profile_Identifier.ipynb

## Getting Started

For getting a copy of the project up and running in local machine follow the steps:

1. Acquire the first Twitter dataset from http://mib.projects.iit.cnr.it  used in paper "The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the Arms Race"
2. Clone the repository 
```
git clone https://github.com/abhishekSen999/Fake-Profile-Identification-OSN.git
```
3. Save the twitter dataset in 
```
"/data/tagged data"
````

### Prerequisites

1. Create a developers account on Twitter and acquire the required credentials to use Twitter Api.
2. Install numpy, matplotlib, pandas, jupyter notebook. (refer to https://scipy.org/install.html)

```
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
```
3. Install SKLearn (refer to https://scikit-learn.org/stable/install.html)
```
sudo pip install -U scikit-learn
```
4. Install  TensorFlow2 (refer to https://www.tensorflow.org/install/)
```
sudo pip install tensorflow
```
5. Install Keras (refer to https://keras.io/)
```
sudo pip install keras

```
6. Install Tweepy(refer to https://github.com/tweepy/tweepy)
```
pip install tweepy
```
7. Install JSON Library.
8. Install pylab 0.0.2 (refer to https://pypi.org/project/pylab/)
```
pip install pylab==0.0.2
```

### Understanding the Contents of the Repository
1. [data_prepocessing.ipynb](data_prepocessing.ipynb) : Run each cell to preprocess the dataset acquired. The desired attributes for the supervised and unsupervised learning models will be extracted in a file "/data/twitter_dataset.csv"

2. [artificial_neural_network.ipynb](artificial_neural_network.ipynb) : Testing Artificial Neural Network on the acquired dataset and tuning of the hyper-parameters is done based on experimentation and its performance is measured.

3. [decision_tree.ipynb](decision_tree.ipynb) : Decision Tree classifier is applied on the acquired dataset to test it's performance.

4. [decision_tree_k_fold.ipynb](decision_tree_k_fold.ipynb) : K_fold cross validation is done on the decision tree model to validate the results.

5. [hierarchical_agglomerative_clustering.ipynb](hierarchical_agglomerative_clustering.ipynb) : The labels of the labeled data is removed and HAC is applied to check the presence of the desired clusters in dendogram.

6. [k_means_clustering.ipynb](k_means_clustering.ipynb) : The labels of the labeled data is removed. PCA is applied to  reduce to two dimentionality. Visualization is done to check presence of clusters and performance of K means Clustering is checked.

7. [k_nearest_neighbors.ipynb](k_nearest_neighbors.ipynb) : K_nearest-neighbors is applied on the acquired data to check it's performance.

8. [naive_bayes_classifier.ipynb](naive_bayes_classifier.ipynb) : K_nearest-neighbors is applied on the acquired data to check it's performance.

9. [rocchio_classifier.ipynb](rocchio_classifier.ipynb) : Rocchio classifier is applied on the acquired dataset to check its performance.

10. [support_vector_machine.ipynb](support_vector_machine.ipynb) : Support Vector Machine is applied on the acquired dataset to check its performance.

The top three algorithms which gives the highest levels of performance are re-coded to form a Hybrid model.

11. [decision_tree.py](decision_tree.py) : Decision tree is re-coded in py script as a part of hybrid model.

12. [k_nearest_neighbors.py](k_nearest_neighbors.py) : K Nearest Neighbors is re-coded in py as a part of hybrid model.

13. [neural_network.py](neural_network.py) : Artificial Neural Network is recoded in py script as a part of hybrid model

14. [hybrid_model.py](hybrid_model.py) : The Hybrid Model created using the above three algorithms is tested and its performance is checked on the acquired dataset.

15. [Fake_Profile_Identifier.ipynb](Fake_Profile_Identifier.ipynb) : Based on the above created Hybrid Model. this is the final identifier. Run each cell to take the screen_name of a profile as input , collect the desired features from Twitter using Twitter API and predict whether profile is Genuine or Fake.



## License

This is to certify that the minor project work entitled “Fake Profile Identification on Online Social Network” has been carried out by **[Megha Sundriyal](https://github.com/meghasundriyal)**  and **[Abhishek Sen](https://github.com/abhisheksen999)** at the Department of Computer Science, University of Delhi under the supervision of Prof. Punam Bedi. This work has been carried out for the partial fulfillment of the requirements of M.Sc. degree in the Department of Computer Science, University of Delhi. This project has not been submitted anywhere else for any other degree or diploma. 

This research work can be reused only for further research by referencing this repository and acknowledging the Authors.


