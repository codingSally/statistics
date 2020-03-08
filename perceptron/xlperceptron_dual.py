import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils.plot import plot_decision_regions
from xlperceptron_base import SLPerceptronBase

class SLPerceptronDual(SLPerceptronBase):
    
    def __init__(self, eta=0.1, n_iter=50):
        super(SLPerceptronDual, self).__init__(eta=eta, n_iter=n_iter)
        
    # 计算Gram Matrix
    def calculate_g_matrix(self, X):
        # 样本数量
        n_samples = X.shape[0] 
        self.G_matrix = np.zeros((n_samples,n_samples))
        # 填充矩阵
        for i in range(n_samples):
            for j in range(n_samples):
                self.G_matrix[i][j] = np.sum(X[i] * X[j])
                
    # 计算判别函数
    def judge(self, X, y, index):
        temp = self.b
        n_samples = X.shape[0]
        
        for _ in range(n_samples):
            temp += self.alpha[_] * y[_] * self.G_matrix[index][_]
            
        return temp * y[index]

    def fit(self, X, y):
        
        # 读取数据集中含有的样本数，特征向量数
        n_samples, n_features = X.shape
        # alpha是一个向量，初始值是0向量； b是一个数值，初始值是0 
        self.alpha, self.b = [0] * n_samples,0
        self.w = np.zeros(n_features)
        
        # 计算Gram Matrix
        self.calculate_g_matrix(X)
        
        i = 0
        while i < n_samples:
            if(self.judge(X,y,i) <= 0):
                self.update(i,y[i])
                i = 0
            else:
                i += 1
        
        for j in range(n_samples):
            self.w += self.alpha[j] * y[j] * X[j]
        
        return self
                     
                
    def update(self, cur_i, y_i):
        self.alpha[cur_i] += self.eta
        self.b += self.eta * y_i
        
def main():
    iris = load_iris()
    X = iris.data[:100, [0, 2]]
    y = iris.target[:100]
    y = np.where(y == 1, 1, -1)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3)
    ppn = SLPerceptronDual(eta=0.1, n_iter=10)
    ppn.fit(X_train, y_train)
    plot_decision_regions(ppn, X, y)


if __name__ == "__main__":
    main()