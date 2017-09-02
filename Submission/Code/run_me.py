import numpy as np
import matplotlib.pyplot as plt
import cluster_utils
import cluster_class
from sklearn.cluster import KMeans

#Load train and test data
train = np.load("../../Data/ECG/train.npy")
test = np.load("../../Data/ECG/test.npy")
cond = (test[:,-1] == 3)
print test[cond].shape
#Create train and test arrays
Xtr = train[:,0:-1]
Xte = test[:,0:-1]
Ytr = np.array(map(int, train[:,-1]))
Yte = np.array( map(int, test[:,-1]))

#Add your code below


'''
Fit Kmeans to test data using 1-40 clusters
'''

quality_scores = np.empty([40])
for k in range(1,41):
	kmeans = KMeans(n_clusters=k, random_state=0).fit(Xtr)
	labels = kmeans.labels_
	quality_scores[k-1] = cluster_utils.cluster_quality(Xtr,labels,k)
optimal = cluster_utils.optimal_cluster_number(quality_scores)


'''
Create figure 1
'''
labels = ["Clusters","Silhouette Score"]
x_axis = np.arange(1,41, dtype=np.int)
plt.figure(1, figsize=(6,4))
plt.plot(x_axis,quality_scores,'or-',linewidth=3)
plt.grid(True)
plt.ylabel(labels[1])
plt.xlabel(labels[0])
plt.title("Clusters vs. Silhouette Score")
plt.xlim(-0.1,40.1)
plt.ylim(-1.1,1.1)
plt.legend(labels,loc="best")
plt.tight_layout()
plt.savefig("../Figures/quality_line_plot.png")

'''
Using optimal cluster number, find cluster proportions
'''
kmeans = KMeans(n_clusters=optimal, random_state=0).fit(Xtr)
labels = kmeans.labels_
proportions = cluster_utils.cluster_proportions(labels,optimal)

'''
Create figure 2, bar chart
'''

x_axis = np.arange(0,optimal, dtype=np.int)
plt.figure(2, figsize=(6,4))
plt.bar(x_axis,proportions, align='center')
plt.grid(True)
plt.ylabel("Proportion")
plt.xlabel("Cluster")
plt.title("Proportion per Cluster")
plt.xlim(-0.75,9.75)
plt.ylim(0,0.15)
plt.tight_layout()
plt.savefig("../Figures/proportion_bar_chart.png")

cluster_means = cluster_utils.cluster_means(Xtr,labels,optimal)
'''
Create figure 3, waveform means
'''

cluster_utils.show_means(cluster_means,proportions).savefig("../Figures/means_waveform.png", bbox_inches='tight')
'''
'''
test_waveforms = [2,4,6,11,16,23,30]
for k in test_waveforms:
	kmeans = KMeans(n_clusters=k, random_state=0).fit(Xtr)
	labels = kmeans.labels_
	proportions = cluster_utils.cluster_proportions(labels,k)
	cluster_means = cluster_utils.cluster_means(Xtr,labels,k)
	cluster_utils.show_means(cluster_means,proportions).savefig("../Figures/Test Waveforms/test_waveform"+str(k)+".png", bbox_inches='tight')

'''
Cluster classifier
'''

accuracy_score = np.empty([40])
for k in range(1,41):
	classifier = cluster_class.cluster_class(k).fit(Xtr,Ytr)
	accuracy_score[k-1] = classifier.score(Xte,Yte)
'''
Create figure 4
'''
labels = ["Clusters","Accuracy Score"]
x_axis = np.arange(1,41, dtype=np.int)
plt.figure(1, figsize=(10,4))
plt.plot(x_axis,accuracy_score,'sb-',linewidth=3)
plt.grid(True)
plt.ylabel(labels[1])
plt.xlabel(labels[0])
plt.title("Clusters vs. Accuracy Score")
plt.xlim(-0.1,40.1)
plt.ylim(-.1,1.1)
plt.gca().set_xticks(np.arange(1,41))
plt.legend(labels,loc="best")
plt.tight_layout()
plt.savefig("../Figures/accuracy_line_plot.png")
