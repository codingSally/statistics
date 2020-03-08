import numpy as np

class SLNaiveBayesBase(object):
    
    def __init__(self):
        pass
    
    # 训练函数 trainMatrix：训练集[是向量化后的文档] trainCategory：文档中每个词条的类别标注
    def fit(self, trainMatrix, trainCategory):
        '''
        朴素贝叶斯分类器训练函数，求：p(Ci),基于词汇表的p(w|Ci)
        Args:
            trainMatrix : 训练矩阵，即向量化表示后的文档（词条集合）
            trainCategory : 文档中每个词条的列表标注
        Return:
            p0Vect : 属于0类别的概率向量(p(w1|C0),p(w2|C0),...,p(wn|C0))
            p1Vect : 属于1类别的概率向量(p(w1|C1),p(w2|C1),...,p(wn|C1))
            p1 : 属于1类别文档的概率
            
        要做分类，那就是要求P(C1|W) 或者  P(C0|W)
        其中 P(C0|W) = P(W|C0)*P(C0)/P(W)
           P(C1|W) = P(W|C1)*P(C1)/P(W)
        由于分母都是一样的，所以可以只求分子，比较分子的大小，即只求：
            P(W|C0)、P(C0)、P(W|C1)、P(C1)
        P(C0)、P(C1) 为先验概率，表示每种类别的分布概率
        P(W|C0)、P(W|C1) 为条件概率，表示某种类别下，某事发生的概率
        P(C0|W)、P(C1|W) 为后验概率，表示某事发生了，它属于某种类别的概率
        '''
        # 样本的数量
        numTrainDocs = len(trainMatrix)
        # 长度为词汇表长度
        numWords = len(trainMatrix[0])
        # 1. 求先验概率： p0 p1 为先验概率，即每种类别的分布概率
        # 1类别文档的概率 p(c1)
        self.p1 = sum(trainCategory) / float(numTrainDocs)
        self.p0 =  1- self.p1
        
        # 2. 求条件概率
        # 由于后期要计算p(w|Ci)=p(w1|Ci)*p(w2|Ci)*...*p(wn|Ci)，若wj未出现，则p(wj|Ci)=0,因此p(w|Ci)=0，这样显然是不对的
        # 故在初始化时，将所有词的出现数初始化为1，分母即出现词条总数初始化为2[为每个特征可取值个数]
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        p0Denom = 2.0
        p1Denom = 2.0
        
        for i in range(numTrainDocs):
            if trainCategory[i] == 1 :
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        # p(wi | c1)
        # 为了避免下溢出（当所有的p都很小时，再相乘会得到0.0，使用log则会避免得到0.0）
        # 根据极大似然估计算得
        self.p1Vect = np.log(p1Num / p1Denom)
        self.p0Vect = np.log(p0Num / p0Denom)
        
        return self
    
    
    # 预测函数
    def predict(self, testX):
        '''
        朴素贝叶斯分类器
        Args:
            testX : 待分类的文档向量（已转换成array）
            p0Vect : p(w|C0)
            p1Vect : p(w|C1)
            p1 : p(C1)
        Return:
            1 : 为侮辱性文档 (基于当前文档的p(w|C1)*p(C1)=log(基于当前文档的p(w|C1))+log(p(C1)))
            0 : 非侮辱性文档 (基于当前文档的p(w|C0)*p(C0)=log(基于当前文档的p(w|C0))+log(p(C0)))
        '''
        predict_p1 = np.sum(testX * self.p1Vect) + np.log(self.p1)
        predict_p0 = np.sum(testX * self.p0Vect) + np.log(self.p0)
        
        if predict_p1 > predict_p0:
            return 1
        else:
            return 0
        
def loadDataSet():
    '''数据加载函数。这里是一个小例子'''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论，代表上面6个样本的类别
    return postingList, classVec


def checkNB():
    '''测试'''
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postDoc in listPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postDoc))

    nb = SLNaiveBayesBase()
    nb.fit(np.array(trainMat), np.array(listClasses))

    testEntry1 = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry1))
    print(testEntry1, 'classified as:', nb.predict(thisDoc))

    testEntry2 = ['stupid', 'garbage']
    thisDoc2 = np.array(setOfWord2Vec(myVocabList, testEntry2))
    print(testEntry2, 'classified as:', nb.predict(thisDoc2))


if __name__ == "__main__":
    checkNB()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        