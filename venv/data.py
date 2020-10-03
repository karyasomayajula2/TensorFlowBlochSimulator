
import numpy as np
import random
from numpy import asarray
from PIL import Image
import datetime
import matplotlib.pyplot as plt
from phantominator import shepp_logan
#imgDpi = 16  # for 62 x 62 pixel image
imgPath = 'C:/Users/aryas/PycharmProjects/MRZero/venv/testImages'
image = Image.open('circles.PNG')
class data:
    def getSheppLogan(self):
        M0, T1, T2 = shepp_logan((32, 32, 1), MR=True, zlims=(-.25, .25));
        return M0;

    def imgCircles(self):
        data = np.asarray(image);
        return data;

    def imgCreator(self, imgarrRow, imgarrCol, minnumDots, maxnumDots, numImages):
        imgArr = np.zeros((imgarrRow, imgarrCol), dtype=int)

        for i in range(0, numImages):
            for j in range(0, random.randint(minnumDots, maxnumDots)):
                imgArr[random.randint(0, imgarrRow - 1)][random.randint(0, imgarrCol - 1)] = random.randint(0, 250)
            plt.figure(i + 1)
            plt.imshow(imgArr)
            plt.gray()
            plt.axis('off')
            T1Value = round(random.uniform(0.3, 6.1), 2);
            T2Value = round(random.uniform(0, 3.8), 2);
            #plt.savefig(imgPath + str(datetime.date.today()) + "_" + str(i + 1) + '.jpg', bbox_inches='tight', dpi=imgDpi)
            imgArr = imgArr;
        return imgArr, T1Value, T2Value;