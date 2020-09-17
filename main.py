from PIL import Image
import numpy as np
import random
from tool import random_delete,normalize,compute_loss


if __name__ == '__main__':

    #mg=mpimg.imread('woman.jpg')
    origin=Image.open('woman.jpg')
    origin=np.array(origin)

    row=origin.shape[0]
    col=origin.shape[1]

    # img2=np.array(img)
    img, content = random_delete(origin)  #随机用平均值遮挡图片部分像素。content矩阵是原图中没被删掉的数的位置为1，其余为0


    k=128
    learning_rate = 0.00002
    epochs=500
    decay_rate=0.9
    decay_steps=150
    u=np.random.randn(row,k)
    v=np.random.randn(k,col)
    # u=normalize(u)
    # v=normalize(v)
    for epoch in range(0,epochs):
        del_u=np.zeros((row,k))
        del_v=np.zeros((k,col))

        for i in range(0,row):
            for j in range(0,col):
                tmp = np.sum(u[i,:] * v[:,j])
                del_u[i,:]=del_u[i,:]-(img[i,j]-tmp)*v[:,j]*content[i,j]
                del_v[:,j]=del_v[:,j]-(img[i,j]-tmp)*u[i,:]*content[i,j]

        # for i in range(0, col):
        #     for j in range(0, row):
        #         tmp=np.sum(v[:,i]*u[j,:])
        #         del_v[:,i] =del_v[:,i] -(img[j,i] - tmp)*u[j,:]*content[j,i]

        u=u-learning_rate*del_u
        v=v-learning_rate*del_v

        learning_rate=learning_rate*pow(decay_rate,int(epoch/decay_steps))
        print("loss=",compute_loss(img,u,v,content))

    img_after=np.dot(u,v)
    img_after = Image.fromarray(img_after)
    img_after.show()



