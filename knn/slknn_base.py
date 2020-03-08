import numpy as np
from math import sqrt
from collections import Counter

class SLKNN(object):
    
    def __init__(self, k=3, p=2):
        """
         k:临近数
         p：距离度量
        """
        self.k = k
        self.p = p
        
     
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    # 预测函数
    def predict(self, X):  # X为当前样本点
        
        # 1. 计算样本点到每个训练集的距离
#         distances = []
#         for _ in X_train:
#             d = sqrt(np.sum((X - _) **2))
#             distances.append(d)
        distances = [sqrt(np.sum((X - x_train) ** 2)) for x_train in self.X_train]
        # 2. 排序
        nearest = np.argsort(distances)
        
        # 3. 选择最近的K个
        top_y = [self.y_train[index] for index in nearest[:self.k]]
        
        # 4. 统计分类的数量
        votes = Counter(top_y)
        
        return votes.most_common(1)[0][0]
    
    # 测试集评估准确率
    def accuracy(self, X_test, y_test):
        right_count = 0
        
        for X,y in zip(X_test,y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count /len(X_test)  
        

if __name__ == "__main__": #如果模块是被直接运行的，则代码块被运行，如果模块是被导入的，则代码块不被运行。
    main(SLKNN) 
        