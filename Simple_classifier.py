# Split train and test dataset and run all classifier


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

def extract_color_stats(image):

	features = [np.mean(image), np.std(image),
		]


	return features


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
	help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
	help="type of python machine learning model to use")
args = vars(ap.parse_args())


models = {
	"knn": KNeighborsClassifier(n_neighbors=1),
	"naive_bayes": GaussianNB(),
	"logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
	"svm": SVC(kernel="linear"),
	"decision_tree": DecisionTreeClassifier(),
	"random_forest": RandomForestClassifier(n_estimators=100),
	"mlp": MLPClassifier(),
	"lda": LinearDiscriminantAnalysis()   
}


#print("[INFO] extracting image features...")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []


for imagePath in imagePaths:

	image = Image.open(imagePath)
	features = extract_color_stats(image)
	data.append(features)


	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)


le = LabelEncoder()
labels = le.fit_transform(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.50)



print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)


print("[INFO] evaluating...")
predictions = model.predict(testX)
print(classification_report(testY, predictions,
	target_names=le.classes_))
results = confusion_matrix(testY, predictions) 
print('confusion_matrix',results)
