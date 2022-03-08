from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from DatasetLoader import DatasetLoader 
from Preprocessor import Preprocessor
from imutils import paths
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pprint

def plot_data(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
     
    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
    
     
    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts 
        counts = counts[sorted_index]
        ylabel_text = 'count'
         
    print(counts)
    xtemp = np.arange(len(unique))
     
    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('Surgery Phases')
    plt.ylabel(ylabel_text)
    

dataset = str(Path("C:/labeledFrames"))

phases = ["1_Incision", "2_ViscousAgentInjection", "3_Rhexis", "4_Hydrodissection", "5_Phacoemulsification",
          "6_IrrigationAndAspiration", "7_CapsulePolishing", "8_LensImplantSettingUp", "9_ViscousAgentRemoval",
          "X_TonifyingAndAntibiotics"]

print("::: [INFO] loading images... :::")
imagePaths = list(paths.list_images(dataset))
pre = Preprocessor(100, 100)
dl = DatasetLoader(preprocessors=[pre])
(data, labels) = dl.load(imagePaths, phases)
print("::: [INFO] size of features matrix: {:.1f}MB :::".format(
	data.nbytes / (1024 * 1024.0)))
le = LabelEncoder()
labels = le.fit_transform(labels)
data = data.reshape((np.size(data, 0), 30000))
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("::: [INFO] Plotting Data Distribution... :::")
plt.suptitle('relative amount of surgery phases')
plot_data(trainY, loc='left', relative=False)
plot_data(testY, loc='right', relative=False)
plt.legend([
    'train ({0} photos)'.format(len(trainY)), 
    'test ({0} photos)'.format(len(testY))
]);

plt.show()
    

print("::: [INFO] evaluating k-NN classifier... :::")



model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(trainX, trainY)
y_pred = model_knn.predict(testX)
print("::: k-NN Report :::")
print(classification_report(testY, y_pred, target_names=le.classes_))
print("::: confusion matrix :::")
print(confusion_matrix(testY, y_pred))

print("[INFO] ::: Saving Model ::::")
joblib.dump(model_knn, "classifier.pkl")

    
print("Done...")
