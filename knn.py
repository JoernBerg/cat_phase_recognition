from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#import joblib
from DatasetLoader import DatasetLoader 
from Preprocessor import Preprocessor
from imutils import paths
from pathlib import Path
import numpy as np



dataset = str(Path("C:/labeledFrames"))

print("::: [INFO] loading images... :::")
imagePaths = list(paths.list_images(dataset))

pre = Preprocessor(100, 100)
dl = DatasetLoader(preprocessors=[pre])
(data, labels) = dl.load(imagePaths, 500)
print("::: [INFO] size of features matrix: {:.1f}MB :::".format(
	data.nbytes / (1024 * 1024.0)))
le = LabelEncoder()
labels = le.fit_transform(labels)
data = data.reshape((np.size(data, 0), 30000))
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("::: [INFO] evaluating k-NN classifier... :::")



model_knn = KNeighborsClassifier(n_neighbors=121)
model_knn.fit(trainX, trainY)
y_pred = model_knn.predict(testX)
print("::: k-NN Report :::")
print(classification_report(testY, y_pred, target_names=le.classes_))
print("::: confusion matrix :::")
print(confusion_matrix(testY, y_pred))

#print("[INFO] ::: Saving Model ::::")
#joblib.dump(model, "classifier.pkl")

    
print("Done...")
input("Press Enter to continue...")