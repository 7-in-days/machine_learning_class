import numpy as np
import matplotlib as plt
import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
import pickle

"""
임의의 x0, x1 값에 대해서 f의 최솟값을 grdeitn descent를
이용하여 구하라

f(x0, x1) = x0^2 + x1^2

"""

# def numerical_diff(f, x):
#     h = 1e-4
#     return (f(x+h)-f(x-h))/(2*h)

def function_1(x):
    return x[0] * x[0] + x[1] * x[1]

# def function_2_1(x0):
#     return x0*x0 + x1*x1 # x1 전역변수 -> 편미분

# def function_2_2(x1):
#     return x1*x1 + x0*x0 # x0 전역변수


def numerical_gradient_no_batch(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h # f(x+h)

        fxh1 = f(x)

        x[idx] = tmp_val - h # f(x-h)
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h) # gredient
        x[idx] = tmp_val # 얘는 왜? 아 정상화 때문에? 기존에 할당할 때 값으로 복귀한다고 생각하면 될 듯

    
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_no_batch(f, X)
    
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_no_batch(f,x)

        return grad

def gradient_descent(f, init_x, lr=0.0001, step_num=100000):
    x = np.array(init_x)
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f,x)

        x -= lr*grad

    return x


x0 = float(input("숫자 x0를 입력하세요: "))
x1 = float(input("숫자 x1을 입력하세요: "))


init_x = np.array([x0, x1]) # 배열 넘기기
low_x = gradient_descent(function_1, init_x)
fx_low = function_1(low_x)

print("초기값:", init_x)
print("최솟값 근사 좌표:", low_x)
print("최솟값 근사 함수값:", fx_low)
