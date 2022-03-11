import cv2 as cv
from pathlib import Path
import joblib
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import numpy as np
import os 

video_path = "C:/Users/Jörn/Documents/FH/BA/cataract-101/videos/case_350.mp4"
    
def process_video(path):
    video_file = str(Path(path))
    capture = cv.VideoCapture(video_file)

    if capture.isOpened():
        ret = True
        count = 0

    
        while ret == True:
            ret, frame = capture.read()
            try:
                cv.imwrite("C:/testFrames/frame%d.jpg" % count, frame)
                if (count%200) == 0:
                    print("::: [INFO] Writing Frames... :::")
                count += 1
            except cv.error:
                break
    else:
        print("::: [ERROR] Could not open file! :::")
        

def write_labeled(y, path):
    switch_dict = {
        0: "1_Incision",
        1: "2_ViscousAgentInjection",
        2: "3_Rhexis",
        3: "4_Hydrodissection",
        4: "5_Phacoemulsification",
        5: "6_IrrigationAndAspiration",
        6: "7_CapsulePolishing",
        7: "8_LensImplantSettingUp",
        8: "9_ViscousAgentRemoval",
        9: "X_TonifyingAndAntibiotics"
    }
    frame = cv.imread(path)
    cv.imwrite("C:/labeledTest/{}/{}".format(switch_dict.get(y), path.split(os.path.sep)[-1]), frame)
    


process_video(video_path)
    
dataset = str(Path("C:/testFrames"))
    
imagePaths = list(paths.list_images(dataset))


print("::: [INFO] Loading data... :::")
data = []
for (i, path) in enumerate(imagePaths):
    if (i%500) == 0:
        print("::: [INFO] still loading... :::")
    img = cv.imread(path)
    img = cv.resize(img, (100, 100))
    data.append(img)

data = np.array(data)
print("::: [INFO] Flatten Pictures to Feature-Vectors... :::")
data = data.reshape((np.size(data, 0), 30000))

model = joblib.load("classifier.pkl")

y_pred = model.predict(data)
print("::: [INFO] classified frames :::")

for i in range(len(y_pred)):
    if (i%200) == 0:
        print("::: [INFO] Sorting Frames... :::")
    write_labeled(y_pred[i], imagePaths[i])


print("::: [INFO] Done... :::")
