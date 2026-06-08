# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from dataset.mnist import load_mnist
# sys.path.append(os.pardir)

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# # parameter

# input_dim = 784
# hidden_dim = 50
# output_dim = 10
# learning_rate = 0.1
# epochs = 200 # 30
# batch_size = 100


# np.random.seed(42)

# # hidden layer 들어가는 파라미터 초기화
# W1 = np.random.randn(input_dim, hidden_dim) * 0.01
# b1 = np.zeros((1, hidden_dim))

# W2= np.random.randn(hidden_dim, hidden_dim) * 0.01 # hidden이랑 output이랑 사이즈 잘 봐야함 안그러면 터짐
# b2 = np.zeros((1, hidden_dim))

# W3 = np.random.randn(hidden_dim, output_dim) * 0.01
# b3 = np.zeros((1, output_dim))



# # activation function

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



# train_size = x_train.shape[0]
# iter_per_epoch = train_size // batch_size

# for epoch in range(epochs):
#         perm = np.random.permutation(train_size) 

#         for i in range(iter_per_epoch):
#             batch_mask = perm[i*batch_size:(i+1)*batch_size]     
#             x_batch = x_train[batch_mask]
#             y_batch = t_train[batch_mask]

#             # forward pass

#             z1 = x_batch @ W1 + b1
#             a1 = sigmoid(z1)
#             #a1 = relu(z1)
#             z2 = a1 @ W2_three + b2_three
#             a2 = sigmoid(z2)
#             #a2 = relu(z2)
#             z3 = a2 @ W3 + b3
#             t = softmax(z3)
#             loss = cross_entropy(t, y_batch)

#             # backward pass - output layer
#             dL_dz3 = (t - y_batch) / batch_size
#             dL_dW3 = a2.T @ dL_dz3
#             dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

#             # backward pass - hidden layer 2
#             dL_da2 = dL_dz3 @ W3.T
#             dL_dz2 = dL_da2 * sigmoid_deriviate(z2)
#             #dL_dz2 = dL_da2 * relu_derivative(z2)
#             dL_dW2 = a1.T @ dL_dz2
#             dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

#             # backward pass - hidden layer 1
#             dL_da1 = dL_dz2 @ W2_three.T
#             dL_dz1 = dL_da1 * sigmoid_deriviate(z1)
#             #dL_dz1 = dL_da1 * relu_derivative(z1)
#             dL_dW1 = x_batch.T @ dL_dz1
#             dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

#             # update parameters

#             W1 -= learning_rate * dL_dW1
#             b1 -= learning_rate * dL_db1
#             W2_three -= learning_rate * dL_dW2
#             b2_three -= learning_rate * dL_db2
#             W3 -= learning_rate * dL_dW3
#             b3 -= learning_rate * dL_db3

#         loss_three_layer_list.append(loss)

#         def accuracy(X, y_true):
#             z1 = X @ W1 + b1
#             a1 = sigmoid(z1)
#             #a1 = relu(z1)
#             z2 = a1 @ W2_three + b2_three
#             a2 = sigmoid(z2)
#             #a2 = relu(z2)
#             z3 = a2 @ W3 + b3
#             t = softmax(z3)
#             y_pred = np.argmax(t, axis=1)
#             y_true = np.argmax(y_true, axis=1)
#             return np.mean(y_pred == y_true)
        
#         acc = accuracy(x_test, t_test)
#         accuracy_three_layer_list.append(acc)