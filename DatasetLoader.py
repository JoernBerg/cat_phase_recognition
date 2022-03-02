import numpy as np
import cv2 as cv
import os

class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        
        if self.preprocessors is None:
            self.preprocessors = []
        
    
    def load(self, ImagePaths, verbose=-1):
        data = []
        labels = []
        
        for (i, imagePath) in enumerate(ImagePaths):
            image = cv.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
                
            data.append(image)
            labels.append(label)
        
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                 print("::: [INFO] processed {}/{} :::".format(i+1, len(ImagePaths)))
        
        return (np.array(data), np.array(labels))
            