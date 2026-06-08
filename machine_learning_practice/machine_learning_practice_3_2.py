import numpy as np
import matplotlib as plt
import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
import pickle


"""
Mnist dataset을 활용하여 batchsize 100으로 sample weight이
있는 것을 활용하여 3layer nn에 적용했을 때 accuracy를 구하는 코드 작성
"""

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    # print(x_train.shape)
    # print(t_train.shape)
    # print(x_test.shape)
    # print(t_test.shape)

    return x_test, t_test

def init_network():
    with open(os.path.dirname(__file__) + "/3_week_sample_weight_minist.pkl", 'rb') as f:
        network = pickle.load(f)
    
    return network


def sigmoid(x): # 시그모이드 함수
    return 1/(1+np.exp(-x))

def softmax(a): # softmax function
    c = np.max(a)
    exp_a = np.exp(a-c) # 너무 큰 값을 방지하기 위한 방법.
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3'], 

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()

network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])

print("Accuracy : " + str(float(accuracy_cnt / len(x))))
