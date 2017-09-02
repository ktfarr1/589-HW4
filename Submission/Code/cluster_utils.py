import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def cluster_quality(X,Z,K):
    '''
    Compute a cluster quality score given a data matrix X (N,D), a vector of 
    cluster indicators Z (N,), and the number of clusters K.
    '''
    # Compute distance matrix to prevent repetition
    distance = pairwise_distances(X,metric='euclidean')
    data_elements = X.shape[0]
    silhouette_array = np.empty(data_elements)
    
    # loop over data matrix, assigning a silhouette score to each data point
    for data_case in range(0,data_elements):
        score = silhouette_score(data_case,X,Z,K,distance)
        silhouette_array[data_case] = score
    
    #compute mean and return
    return np.mean(silhouette_array)

def silhouette_score(index,data,labels,k,distance):
    '''
    Compute the silhouette score of a given point given the point
    
    '''
    #compute the average distance between the data point given by the index and the rest of the cluster it resides in
    average = average_distance(index,data,labels,k,distance)
    nearest = nearest_cluster_average(index,data,labels,k,distance)
    return (nearest - average)/max(average,nearest)

def average_distance(index,data,labels,k,distance):
    '''
    Compute the average distance within a cluster
        index: int corresponding to index of data item
        data: data matrix
        labels: array of cluster assignments
        k: number of clusters (not used currently)
        distance: pre-computed pairwise distance matrix
    '''
    #Pull the indices corresponding to the cluster
    cluster = np.where(labels == labels[index])[0]
    
    #If a cluster contains zero or one elements, then return the distance as 0 to prevent error
    if len(cluster) == 0 or len(cluster) == 1:
        return 0.0
    
    #loop through the other cluster datum, leaving out the current data point, and calculate the mean distance
    average = np.mean([distance[index,j] for j in cluster if not index==j])
    return average

def nearest_cluster_average(index,data,labels,k,distance):
    #Pull the indices of points in all other clusters
    other_clusters = np.where(labels != labels[index])[0]

    #return 0 if there are no other clusters containing data
    if len(other_clusters) == 0:
        return 0.0

    #Calculate the minimum distance, and then pull the cluster label corresponding to it
    min = np.amin(distance[index,other_clusters])
    nearest_cluster = np.where(distance[index,]==min)[0]

    #Handle no nearest cluster
    if len(nearest_cluster) == 0:
        return 0.0

    #Calculate the mean distance between neighboring cluster and data point
    average = np.mean([distance[index,j] for j in nearest_cluster if not index == j])
    return average

def optimal_cluster_number(scores):
    return np.argmax(scores)+1

def cluster_proportions(Z,K):
    '''
    Compute the cluster proportions p such that p[k] gives the proportion of
    data cases assigned to cluster k in the vector of cluster indicators Z (N,).
    The proportion p[k]=Nk/N where Nk are the number of cases assigned to
    cluster k. Output shape must be (K,)
    '''
    #Initialize empty array for the proportions
    proportions = np.empty(K)

    #Total number of data cases
    cases = len(Z)

    #Loop over each cluster label, pulling out the indices of the cluster, and calculate the average
    for cluster in range(K):
        c = len(Z[Z == cluster])
        proportions[cluster] = c/float(cases)
    return proportions
        
def cluster_means(X,Z,K):
    '''
    Compute the mean or centroid mu[k] of each cluster given a data matrix X (N,D), 
    a vector of cluster indicators Z (N,), and the number of clusters K.
    mu must be an array of shape (K,D) where mu[k,:] is the average of the data vectors
    (rows) that are assigned to cluster k according to the indicator vector Z.
    If no cases are assigned to cluster k, the corresponding mean value should be zero.
    '''
    #Initialize empty mu array
    D = X.shape[1]
    mu = np.zeros((K,D))

    #Loop over each cluster, and take the mean over the columns
    for cluster in range(K):
        cluster_data = X[Z == cluster]
        if len(cluster_data) > 0:
            mean = np.mean(cluster_data,axis=0)
            mu[cluster] = mean
    return mu
    
def show_means(mu,p):
    '''
    Plot the cluster means contained in mu sorted by the cluster proportions 
    contained in p.
    '''
    K = p.shape[0]
    f = plt.figure(figsize=(8,8))
    for k in range(K):
        plt.subplot(8,5,k+1)
        plt.plot(mu[k,:])
        plt.title("Cluster %d: %.3f"%(k,p[k]),fontsize=5)
        plt.gca().set_xticklabels([])
        plt.gca().set_xticks([25,50,75,100,125,150,175])
        plt.gca().set_yticklabels([])
        plt.gca().set_yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
        plt.ylim(-0.2,1.2)
        plt.grid(True)
    plt.tight_layout()
    return f
        
        
        
    