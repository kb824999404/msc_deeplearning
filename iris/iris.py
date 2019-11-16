import math

from HelperClass.NeuralNet_1_2 import *
from HelperClass.Visualizer_1_0 import *
file_name='iris.txt'
test_file_name='iris_test.npz'
train_file_name='iris_train.npz'

def getIris(file):
    s = [];   f=open(file,'r')
    s=f.readlines();  data=[]
    for row in s:
        oneRow=row.split(',')
        IrisClass={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
        if oneRow[-1][-1]=='\n':
            oneRow[-1]=oneRow[-1][:-1]
        oneRow[-1]=IrisClass[oneRow[-1]]
        oneData=list(map(float, oneRow))
        data.append (oneData)
    data=np.array(data)
    return data 

def generateTestData():
    data=getIris(file_name)
    X=data[:,0:-1]
    Y=data[:,-1:]
    testX=X[::8]               #测试样本，间隔8取一个
    testY=Y[::8]
    trainX=np.delete(X,np.arange(X.shape[0])[::8],axis=0)         #去掉测试样本得到训练样本
    trainY=np.delete(Y,np.arange(Y.shape[0])[::8],axis=0)
    print(testX.shape)
    print(trainX.shape)
    np.savez(train_file_name,data=trainX,label=trainY)
    np.savez(test_file_name,data=testX,label=testY)

def Test(net):
    reader = DataReader_1_3(test_file_name)
    reader.ReadData()
    reader.ToOneHot(num_category, base=1)
    reader.NormalizeX()
    x=reader.XTrain
    y=reader.YTrain
    a=net.inference(x)
    m = a.shape[0]
    ra = np.argmax(a, axis=1)
    ry = np.argmax(y, axis=1)
    r = (ra == ry)
    accuracy = r.sum()/m*100
    print("Accuracy:",accuracy,"%")

# 主程序
if __name__ == '__main__':
    # generateTestData()
    num_category = 3
    reader = DataReader_1_3(train_file_name)
    reader.ReadData()
    reader.ToOneHot(num_category, base=1)
    reader.NormalizeX()

    num_input = 4
    params = HyperParameters_1_1(num_input, num_category, eta=2, max_epoch=5000, batch_size=30, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet_1_2(params)
    net.train(reader, checkpoint=1)

    Test(net)


