from PIL import Image
import numpy as np
import os
from scipy.ndimage import gaussian_filter

dataset = 'CHASE'
testflag = 'train'
file_dir = "/root/vessel-seg/data/{}/{}/".format(dataset,testflag)
img_dir = os.path.join(file_dir, "images")
filenames = [fname for fname in os.listdir(img_dir) if fname.endswith(".jpg")]
filenames = sorted(filenames)
print(filenames)


for file_index in range(len(filenames)):
    img = np.array(Image.open(os.path.join(img_dir,filenames[file_index])))
    print(img.shape)
    greenImg = img[:,:,1]
    for sigma in [1,2,4,8,16]:
        for order in [0,(1,0),(0,1),(2,0),(0,2),(1,1)]:
            output = Image.fromarray(gaussian_filter(greenImg, sigma=sigma, order=order))
            output_dir = "{}features/{}_{}_{}.jpg".format(file_dir,filenames[file_index][:-4],sigma,order)
            output.save(output_dir)
