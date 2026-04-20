import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.functions import *
from common.gradient import numerical_gradient 
import pickle
from dataset.mnist import load_mnist 
import matplotlib.pyplot as plt

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        
        self.params = {} # 딕셔너리 형태?
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1 # wx + b = a
        z1 = sigmoid(a1) # a를 sigmoid 함수에 통과시켜 z1을 구한다. z1은 은닉층의 출력값이 된다.
        a2 = np.dot(z1, W2) + b2 # w2*z1 + b2 = a2
        y = softmax(a2) # softmax

        # a1 -> z1 - > a2 -> y

        return y
    
    # x: 입력 데이터, t: 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0]) # 총 맞춘 개수 / 총 데이터 개수 -> 1보다 작을 듯?
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        grads = {}

        batch_num = x.shape[0] # 배치

        # forward

        a1 = np.dot(x, W1) + b1 # W1 * x + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2 # W2 * z1(hidden output) + b2
        y = softmax(a2)


        # backward (역전파?)

        dy = (y-t) / batch_num # 미분값 / 배치사이즈
        grads['W2'] = np.dot(z1.T, dy) # 이건 맞고
        grads['b2'] = np.sum(dy, axis=0) # 오차값들을 더해서 bias로 재지정?

        da1 = np.dot(dy, W2.T) # 오차 * 가중치
        dz1 = sigmoid_grad(a1) * da1 # sigmoid 역전파 함수 * 미분값?

        grads['W1'] = np.dot(x.T, dz1) # input, 
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
    


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 3000 # 1 에포크당 반복 수
train_size = x_train.shape[0]
batch_size = 200
learning_rate = 0.05 # 학습률


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# mnist는 사이즈가 28*28이기에 input size가 784여야함

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # 임의 추출

    x_batch = x_train[batch_mask] # train set에서 임의로 정한 batch 데이터를 뺀다.

    t_batch = t_train[batch_mask] # 정답도 마찬가지

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # network 내부의 key에 접근 후, learning rate와 gradient를 곱하고 빼서 파라미터 업데이트
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    

    train_acc = network.accuracy(x_train, t_train) # train과 test의 각각 accuracy 업데이트
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print("돌아간 횟수",i)
    


x = np.arange(len(train_loss_list))
train_len = np.arange(len(train_acc_list))




plt.plot(x, train_loss_list, label="train loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.xlim(0, iters_num)

plt.show()


plt.plot(train_len, train_acc_list, label="train acc")
plt.plot(train_len, test_acc_list, label="test acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)


plt.show()


