from functools import reduce

__doc__ = 'description'
__author__ = '13314409603@163.com'

class Perceptron(object):
    def __init__(self,input_num,activator):
        # 初始化感知器，设置输入参数的个数，以及激活函数。
        # 激活函数的类型为double->doubl
        self.activator = activator
        #权重向量初始化为0
        self.weights = [0.0 for i in range(input_num)]
        #偏执项初始化为0.0
        self.bias = 0.0

    def __str__(self):
        #打印学习到的权重、偏执项
        return 'weights\t:%s\nbias\t:%f\n' % (list(self.weights),self.bias)

    def predict(self,input_vec):
        #输出向量，输出感知机的计算结果

        #把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包一起
        #变参[(x1,w1),(x2,w2),(x3,w3),...]
        #然后利用map函数计算[x1*w1,x2*w2,x3*w3...]
        #最后利用reduce求和
        return self.activator(reduce(lambda a,b:a+b,
               map(lambda x_w:x_w[0]*x_w[1] ,zip(input_vec,self.weights)),0.0)+self.bias)

    def train(self,input_vecs,lables,iteration,rate):
        #输入训练数据：一组向量，对应的lable，训练轮数，学习率
        for i in range(iteration):
            self._one_iteration(input_vecs,lables,rate)

    def _one_iteration(self,input_vecs,lables,rate):
        #一次迭代
        samples = zip(input_vecs,lables)
        for (input_vec,lable) in samples:
            output = self.predict(input_vec)
            self._updata_weight(input_vec,output,lable,rate)

    def _updata_weight(self,input_vec,output,lable,rate):
        #感知器更新权重规则：wi <- wi+rate*(output-lable)*input_vec
        delta = lable - output
        self.weights = list(map(
            lambda x_w:x_w[1]+rate*delta*x_w[0],
            zip(input_vec,self.weights)))
        self.bias += rate*delta
def f(x):
    #激活函数
    return 1 if x>0 else 0
def get_training_dataset():
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]
    lables = [1,0,0,0]
    return input_vecs,lables
def train_and_perceptron():
    p = Perceptron(2,f)
    input_vecs,lables = get_training_dataset()
    p.train(input_vecs,lables,20,0.1)
    return p

if __name__ == '__main__':
    perception = train_and_perceptron()
    print(perception)
    print(perception.predict([1, 1]))
    print(perception.predict([1, 0]))
    # a = [1,2,3,4]
    # b = [1.0,2.0,3.0,4.0]
    # print(reduce(lambda a,b:a+b,map(lambda ab: ab[0] * ab[1], zip(a, b))))