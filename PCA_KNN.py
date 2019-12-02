# large dataset is vertorized and detect using knn
#run PCA_KNN.py
#from sklearn.metrics import accuracy_score
#accuracy_score(label_p, predictions)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import scipy.linalg as la






ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="Gallery",
	help="path to directory containing the 'Gallery' dataset")

ap.add_argument("-d1", "--dataset1", type=str, default="Probe",
	help="path to directory containing the 'Probe' dataset1")


ap.add_argument("-m", "--model", type=str, default="knn",
	help="type of python machine learning model to use")
args = vars(ap.parse_args())


models = {
	"knn": KNeighborsClassifier(n_neighbors=1),

}



imagePaths = paths.list_images(args["dataset"])
X_data = []
labels = []


for imagePath in imagePaths:

	image = Image.open(imagePath)
	image=np.array(image)
	X_data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
X_data=np.array(X_data)
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1]*X_data.shape[2])
X_mean=np.mean(X_data)
X_data=X_data-X_mean
X_data= np.array(X_data)
pca = PCA(.80)
principalComponents1 = pca.fit_transform(X_data)
le = LabelEncoder()
labels = le.fit_transform(labels)
print(labels)










print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]

model.fit(X_data,labels)





imagePaths1 = paths.list_images(args["dataset1"])
X_data1 = []
label_p = []

for imagePath1 in imagePaths1:

	image1 = Image.open(imagePath1)
	image1 = np.array(image1)

	X_data1.append(image1)
	label1 = imagePath1.split(os.path.sep)[-2]
	label_p.append(label1)
X_data1=np.array(X_data1)
X_data1=X_data1.reshape(X_data1.shape[0],X_data1.shape[1]*X_data1.shape[2])
X_mean1=np.mean(X_data1)
X_data1=X_data1-X_mean1
X_data1= np.array(X_data1)
pca = PCA(.80)
principalComponents2 = pca.fit_transform(X_data1)
print(principalComponents2.shape)
le = LabelEncoder()
label_p = le.fit_transform(label_p)
   
predictions = model.predict(X_data1)    
print(predictions)    
    
    




