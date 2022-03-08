import numpy as np
import cv2 as cv
import os
import random

class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        
        if self.preprocessors is None:
            self.preprocessors = []
            
    
    def load(self, ImagePaths, phases):
        data = []
        labels = []
        random.shuffle(ImagePaths)
        for phase in phases:
            phase_paths = []
            counter = 0
            for j in range(len(ImagePaths)):
                if counter == 9000:
                    break
                if ImagePaths[j].split(os.path.sep)[-2] == phase:
                    phase_paths.append(ImagePaths[j])
                    counter += 1
            for (i, phase_path) in enumerate(phase_paths):
                label = phase_path.split(os.path.sep)[-2]
                image = cv.imread(phase_path)
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                if (i%200) == 0:
                    print("::: [INFO] Processed Image of Phase {} :::".format(phase))   
                    print(phase_path) 
                data.append(image)
                labels.append(label)
        return (np.array(data), np.array(labels))
            