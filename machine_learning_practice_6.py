# machine_learning_practice_6

# MNIST, 3layer system
# SGD, Momentum, AdaGrad, Adam을 optimizer로 활용하여 각각의 loss 그래프 및 정확도 그리고 비교

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
sys.path.append(os.pardir)
from practice_4.common.multi_layer_net import MultiLayerNet

# data loading
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# parameter

input_dim = 784
hidden_dim = 50
output_dim = 10
learning_rate = 0.1
epochs = 200 # 30
batch_size = 100


np.random.seed(42)

# # hidden layer 들어가는 파라미터 초기화
# W1 = np.random.randn(input_dim, hidden_dim) * 0.01
# b1 = np.zeros((1, hidden_dim))

# W2= np.random.randn(hidden_dim, hidden_dim) * 0.01 # hidden이랑 output이랑 사이즈 잘 봐야함 안그러면 터짐
# b2 = np.zeros((1, hidden_dim))

# W3 = np.random.randn(hidden_dim, output_dim) * 0.01
# b3 = np.zeros((1, output_dim))

# activation function

# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return (x>0.0).astype(float)

# def sigmoid(x):
#     return 1/(1 + np.exp(-x))

# def sigmoid_deriviate(x):
#     s = sigmoid(x)
#     return s * (1-s)

# # 소프트맥스 정규화

# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# # loss function 

# def cross_entropy(preds, targets):
#     return -np.sum(targets * np.log(preds + 1e-9)) / preds.shape[0]



train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size



# 분모 0 방지
epsilon = 1e-7

# SGD


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] # 기존 경사하강법

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None # 물체의 속도

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val) # 매개변수와 같은 구조의 데이터를 딕셔너리 변수로 저장

        
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

# AdaGrad : 개별 매개변수에 적응적으로 학습률을 조정. 즉 학습률을 점차 줄여가는 방법(learning rate decay)
# 과거의 기울기를 제곱하여 계속 더해간다. 그래서 학습 진행 할 수록 갱신 강도가 약해진다.

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}

            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+epsilon)



class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1

        for key in params.keys():
            # 1차/2차 모멘트 갱신
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.iter)
            v_hat = self.v[key] / (1 - self.beta2 ** self.iter)

            # 파라미터 업데이트
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)



optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
accuracy = {}

# optimizer들의 함수 이름을 update로 통일... 딕셔너리 형태로 묶어서 for문 돌릴 수 있음.
# 딕셔너리의 value값 자체를 리스트로 만드는 방법

for key in optimizers.keys():
    networks[key] = MultiLayerNet(input_size=input_dim, hidden_size_list=[hidden_dim, hidden_dim, hidden_dim], output_size=output_dim)
    # 각각 network key에 multilayer 클래스 넣음
    train_loss[key] = []
    accuracy[key] = []


for i in range(epochs):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch) # multilayernet 내부에 내장
        optimizers[key].update(networks[key].params, grads) # key -> optimizer 이름, update는 함수. params는 내장되어 있음(W,b).grads는 경사하강된거

        loss = networks[key].loss(x_batch, t_batch) 
        train_loss[key].append(loss)
        ac = networks[key].accuracy(x_batch, t_batch) 
        accuracy[key].append(ac)


markers = {"SGD" : "o", "Momentum" : "x", "AdaGrad" : "s", "Adam" : "D"}
x = np.arange(epochs)

for key in optimizers.keys():
    plt.plot(x, train_loss[key], marker=markers[key], label=key)
    
plt.xlabel("epochs")
plt.ylabel("loss")
#plt.ylim(0, 1)
plt.legend()
plt.show()

for key in optimizers.keys():
    plt.plot(x, accuracy[key], marker=markers[key], label=key)


plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.legend()
plt.show()