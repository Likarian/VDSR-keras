import glob
import numpy as np
import scipy
import random

class DataGenerator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.LR = []
        self.HR = []

    def flow_from_directory(self, directory, batch_size, iteration):
        image_list = sorted(glob.glob(directory + '/*'))
        while True:
            random.shuffle(image_list)
            for itr in range(iteration):
                for batch_count in range(batch_size):
                    BaseImage = scipy.misc.imread(image_list[itr*batch_size + batch_count])
                    Height, Width = BaseImage.shape[0], BaseImage.shape[1]
                    HighResolutionImage = BaseImage.copy()
                    ResizedBaseImage = scipy.misc.imresize(BaseImage, 1/3, interp='bicubic')
                    LowResolutionImage = scipy.misc.imresize(ResizedBaseImage, (Height, Width), interp='bicubic')
                    
                    self.LR.append(LowResolutionImage)
                    self.HR.append(HighResolutionImage)
                
                inputs = np.asarray(self.LR, dtype=np.float32)
                targets = np.asarray(self.HR, dtype=np.float32)
                self.reset()
                yield inputs, targets