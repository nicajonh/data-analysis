import numpy as np

def Jaccard(a,b):
    """
    自定义杰卡德相似系数函数,仅对0-1矩阵有效,适用于喜欢,收藏等喜好场景相似度计算
    :param a:
    :param b:
    :return:
    """
    return 1.0*(a*b).sum/(a+b-a*b).sum()


class Recommonder(object):

    sim=None

    def similarity(self,x,distance):#计算相似矩阵的函数
        y=np.ones(len(x),len(x))
        for i in range(len(x)):
            for j in range(len(x)):
                y[i,j]=distance(x[i],y[j])
        return y

    def fit(self,x,distance=Jaccard):
        #训练
        self.sim=self.similarity(x,distance)

    def recommend(self,a):
        #推荐
        return np.dot(self.sim,a)*(1-a)
