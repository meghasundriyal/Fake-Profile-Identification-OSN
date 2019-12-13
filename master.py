import pca as my_pca
import naive_bayes as my_naive_bayes
import k_nearest_neighbors as my_k_nearest_neighbors
import support_vector_machine as my_support_vector_machine

dataset_location='data/twitter_dataset.csv'
new_dataset_name_begining='data/twitter_dataset'
new_dataset_extension='.csv'
pca_components_range=range(2,6)

for pca_components in pca_components_range:
    new_name= new_dataset_name_begining+ str(pca_components) +new_dataset_extension
    my_pca.pca(dataset_location, new_name ,pca_components)


for pca_components in pca_components_range:
    new_name=new_dataset_name_begining+ str(pca_components) +new_dataset_extension
    print('====================================================================================================')
    print('----------------------------------',new_name,'-------------------------------------')
    print('====================================================================================================')
    my_naive_bayes.naive_bayes(new_name)
    my_k_nearest_neighbors.k_nearest_neighbours(new_name)
    # my_support_vector_machine.support_vector_machine(new_name)



# non reduced data
print('====================================================================================================')
print('----------------------------------',dataset_location,'-------------------------------------')
print('====================================================================================================')
    
my_naive_bayes.naive_bayes(dataset_location)
my_k_nearest_neighbors.k_nearest_neighbours(dataset_location)
# my_support_vector_machine.support_vector_machine(dataset_location)







