# machine learning practice 2
# (A) activation function을 sigmoid로
# (B) activation function을 ReLU로
# 참고로 3층으로 해야함(layer 3개?)

import numpy as np
import matplotlib.pyplot as plt


# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7 # 가중치와 임계값 설정
#     tmp = w1*x1 + w2*x2
#     if tmp <= theta:
#         return 0
#     else:
#         return 1 
    
# # XOR 게이트는 AND, NAND 게이트로 표현 가능
# def NAND(x1, x2):
#     w1, w2, theta = -0.5, -0.5, -0.7 # 가중치와 임계값 설정
#     tmp = w1*x1 + w2*x2
#     if tmp <= theta:
#         return 0
#     else:
#         return 1

# def OR(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.2 # 가중치와 임계값 설정
#     tmp = w1*x1 + w2*x2
#     if tmp <= theta:
#         return 0
#     else:
#         return 1

# def XOR(x1, x2):
#     return AND(NAND(x1, x2), OR(x1, x2))


Activation_function = "sigmoid" # "relu"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x): # 비선형성 추가
    return np.maximum(0, x)

def step_function(x): # 계단 함수
    if x > 0:
        return 1
    else:
        return 0


def input_layer(x1, x2): # 리턴값 3개
    
    w1 = np.array([0.1, 0.3, 0.5]) # 뉴런 3개
    #print(w1.ndim)
    w2 = np.array([0.2, 0.4, 0.6])

    bias = np.array([0.3, 0.2, 0.1])

    a1 = w1[0]*x1 + w2[0]*x2 + bias[0]
    a2 = w1[1]*x1 + w2[1]*x2 + bias[1]
    a3 = w1[2]*x1 + w2[2]*x2 + bias[2]


    if Activation_function == "sigmoid":
        return sigmoid(a1), sigmoid(a2), sigmoid(a3)
    elif Activation_function == "relu":
        return relu(a1), relu(a2), relu(a3)
    

def hidden_layer(z1, z2, z3):

    w1 = np.array([0.1, 0.3])
    w2 = np.array([0.2, 0.4])
    w3 = np.array([0.3, 0.5])

    bias = np.array([0.1, 0.2])

    a1 = w1[0]*z1 + w2[0]*z2 + w3[0]*z3 + bias[0]
    a2 = w1[1]*z1 + w2[1]*z2 + w3[1]*z3 + bias[1]

    if Activation_function == "sigmoid":
        return sigmoid(a1), sigmoid(a2)
    elif Activation_function == "relu":
        return relu(a1), relu(a2)
    
def output_layer(z1, z2):
    
    w1 = np.array([0.1, 0.2])
    w2 = np.array([0.3, 0.4])

    bias = np.array([0.5, 0.1])

    a1 = w1[0]*z1 + w2[0]*z2 + bias[0]
    a2 = w1[1]*z1 + w2[1]*z2 + bias[1]

    if Activation_function == "sigmoid":
        return sigmoid(a1), sigmoid(a2)
    elif Activation_function == "relu":
        return relu(a1), relu(a2)
    
def main():
    x = np.array([1, 2]) # 임의 값
    z1, z2, z3 = input_layer(x[0], x[1])
    print("z1 : ", z1, "z2 : ", z2, "z3 : ", z3)

    h1, h2 = hidden_layer(z1, z2, z3)
    print("h1 : ", h1, "h2 : ", h2)

    y1, y2 = output_layer(h1, h2)
    print("y1 : ", y1, "y2 : ", y2)

if __name__ == "__main__":
    main()