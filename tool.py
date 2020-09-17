from PIL import Image
import numpy as np
import random
def normalize(a):

    max=np.max(a)
    min=np.min(a)

    a=(a-min)/(max-min)
    return a

def random_delete(img):
    row=img.shape[0]
    col=img.shape[1]
    avg = img.mean()
    content=np.ones((row,col))

    for i in range(0, row):
        for j in range(0, col):
            random_num = random.randint(0, 1000)
            if random_num > 996:
                for p in range(0, 10):
                    for q in range(0, 10):
                        if i + p < row and j + q < col:
                            img[i + p][j + q] = avg
                            content[i + p][j + q] = 0

    return img,content

def compute_loss(R,u,v,content):

    loss=0
    for i in range(0,512):
        for j in range(0,512):
            tmp=np.sum(u[i, :] * v[:, j])
            loss=loss+(R[i,j]-tmp)*(R[i,j]-tmp)*content[i,j]/2

    return loss