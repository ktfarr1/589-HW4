import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

class cluster_class:
    
    def __init__(self,K):
        '''
        Create a cluster classifier object
        '''
        self.K=K #
        self.model = KMeans(n_clusters=K, random_state=0)
        self.clusters_to_labels = np.zeros(K)

    def fit(self, X,Y):
        '''
        Learn a cluster classifier object
        '''
        #cluster X
        self.model.fit(X)
        #compute most frequent label per cluster
        cluster_indicator = self.model.labels_
        for cluster in range(0,self.K):
            labels = Y[cluster_indicator == cluster]
            #no cases, select at random
            if len(labels) == 0:
                self.clusters_to_labels[cluster] = np.random.choice(Y)
            #tie or single option, select between at random
            else:
                #count labels into bins, then extract the index or indices containing the max value
                max_vals = np.where(np.bincount(labels) == np.amax(np.bincount(labels)))[0]
                #Choose at random if there are multiple options, otherwise choose the only option
                self.clusters_to_labels[cluster] = np.random.choice(max_vals)
  
        return self
        
    def predict(self, X):
        '''
        Make predictions usins a cluster classifier object
        '''        
        predictions = np.zeros(X.shape[0])
        cluster_indicator = self.model.predict(X)

        #Loop over each item in the cluster_indicator array, assign the corresponding class label
        for element in range(X.shape[0]):
            predictions[element] = self.clusters_to_labels[cluster_indicator[element]]

        return predictions
    
    def score(self,X,Y):
        '''
        Compute prediction error rate for a cluster classifier object
        '''          
        Yhat = self.predict(X)
        return 1-accuracy_score(Y,Yhat)
        