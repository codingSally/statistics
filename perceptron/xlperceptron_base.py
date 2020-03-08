import numpy as np
from sklearn.model_selection import train_test_split
from utils.plot import plot_decision_regions

class SLPerceptronBase(object):
    
    def __init__(self, eta=0.1, n_iter=50):
        # 步长【学习率】
        self.eta = eta
        # 迭代次数
        self.n_iter = n_iter
        
    def fit(self, X, y):
        # 初始化参数w,b
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        for _ in range(self.n_iter):
            # 统计误分类数
            count = 0;
            for xi, yi in zip(X, y):
                model_pre = self.predict(xi)
#                 model_pre = self.predict(self, xi) //错误的调用形式，调用函数的时候，self本身不用传入
                if yi * model_pre <= 0 :
                    count += 1
                    self.update(xi, yi)
            if count == 0 :
                break
        return self
                
    
    def sign(self, xi):
        return np.dot(xi,self.w) + self.b
    
    def predict(self, xi):
        return np.where(self.sign(xi) <= 0.0 ,-1, 1)
    
    def update(self, xi, yi):
        temp = self.eta * yi * xi
        # 调整temp的形状不变
        temp = temp.reshape(self.w.shape)
        
        self.w += temp
        self.b += self.eta * yi
        
def main():
    iris = load_iris()
    X = iris.data[:100, [0, 2]]
    y = iris.target[:100]
    y = np.where(y == 1, 1, -1)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3)
    ppn = SLPerceptronBase(eta=0.1, n_iter=10)
    ppn.fit(X_train, y_train)
    plot_decision_regions(ppn, X, y)


if __name__ == "__main__":
    main()