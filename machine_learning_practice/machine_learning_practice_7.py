# machine_learning_practice_7
# 가중치의 초깃값

"""
가중치 감소(weight decay) - 오버피팅 방지
초기값을 0으로 하면 모든 가중치의 값이 똑같이 갱신되기 때문에 초깃값을 무작위로 설정해야함.

activation functiond의 분포
가중치가 고루 퍼지지 않으면, 기울기 소실(gradient vanishing)

activation값이 치우쳐지면, 표현력이 제한된다. 
.즉 다수의 뉴런이 거의 같은 값을 출력하고 있다라고 보면 된다.

배치 정규화는 학습 시에 평균과 분산을 신경망 내부에서 조정
학습 과정에서 각 배치 단위별로 데이터가 다양한 분포를 가지더라도 각 배치별로 평균과 분사을 이용하여 정규화하는 것을 의미.
각 층에서 활성화 값이 적당히 분포되도록 조정하는 것 = 배치 정규화 아이디어

"""

# Xavier 초깃값 -> 활성화 함수가 선형이라는 전제로 사용. 각 층의 활성화 분포가 그래도 고루 퍼지게 함. 
# 단, 활성화 함수에 따라 다를 수 있음(tanh가 낫다)
# np.sqrt(1/n)


# He 초기값 : Relu에 특화.
# np.sqrt(2/n) 이는 relu에서 음의 영역이 0이라서 더 넓게 분포시키기 위함?

# dropout : 뉴런 끄고 키고

# 실습 : numpy 활용하여 MNIST 학습, 3 layer network, Batch Normalization
# activation function : relu

# relu니까 he 초기값 기반으로?


import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

# data loading
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


input_size = 784
hidden_size = 50
output_size = 10
learning_rate = 0.1
epochs = 30 # 30
batch_size = 100
weight_decay_lambda = 1e-4

np.random.seed(42)
# hidden layer 들어가는 파라미터 초기화
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))

# output layer 들어가는 파라미터 초기화
W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, hidden_size))

W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b3 = np.zeros((1, output_size))

# batch normalization parameters

gamma1 = np.ones((1, hidden_size))
beta1 = np.zeros((1, hidden_size))
running_mean1 = np.zeros((1, hidden_size))
running_var1 = np.ones((1, hidden_size))

gamma2 = np.ones((1, hidden_size))
beta2 = np.zeros((1, hidden_size))
running_mean2 = np.zeros((1, hidden_size))
running_var2 = np.ones((1, hidden_size))
epsilon = 1e-7


def relu(x):
    return np.maximum(0,x)


def relu_derivative(x):
    return (x>0).astype(np.float32)

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward_inference(x):
    z1 = np.dot(x, W1) + b1
    z1_norm = (z1 - running_mean1) / np.sqrt(running_var1 + epsilon)
    a1 = relu(gamma1 * z1_norm + beta1)
    z2 = np.dot(a1, W2) + b2
    z2_norm = (z2 - running_mean2) / np.sqrt(running_var2 + epsilon)
    a2 = relu(gamma2 * z2_norm + beta2)
    z3 = np.dot(a2, W3) + b3
    return softmax(z3)

# loss function + l2 normalization
def cross_entropy_loss(y, t, weight_decay_lambda):
    m = y.shape[0]
    data_loss = -np.sum(t*np.log(y + 1e-7)) / m

    # l2 정규화 추가
    weight_decay_loss = (weight_decay_lambda / 2) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    return data_loss + weight_decay_loss

# 파라미터 갱신 함수(Weight Decay 적용)

def update_parameters(dW1, db1, dW2, db2, dW3, db3, learning_rate, weight_decay_lambda):
    global W1, b1, W2, b2, W3, b3

    W1 -= learning_rate * (dW1 + weight_decay_lambda * W1) # L2 정규화 추가
    b1 -= learning_rate * db1

    W2 -= learning_rate * (dW2 + weight_decay_lambda * W2) # L2 정규화 추가
    b2 -= learning_rate * db2

    W3 -= learning_rate * (dW3 + weight_decay_lambda * W3) # L2 정규화 추가
    b3 -= learning_rate * db3



train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size

train_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(epochs):

    perm = np.random.permutation(train_size)

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[perm[i:i+batch_size]]
        t_batch = t_train[perm[i:i+batch_size]]
        # forward pass
        z1 = np.dot(x_batch, W1) + b1
        batch_mean  = np.mean(z1, axis=0, keepdims=True)
        batch_var = np.var(z1, axis=0, keepdims=True)

        # 평균과 분산을 이용하여 정규화
        running_mean1 = 0.9 * running_mean1 + 0.1 * batch_mean
        running_var1 = 0.9 * running_var1 + 0.1 * batch_var

        # 정규화된 z1 계산
        z1_norm = (z1 - batch_mean) / np.sqrt(batch_var + epsilon)
        batch_normalized_output = gamma1 * z1_norm + beta1

        a1 = relu(batch_normalized_output)

        z2 = np.dot(a1, W2) + b2
        batch_mean2 = np.mean(z2, axis=0, keepdims=True)
        batch_var2 = np.var(z2, axis=0, keepdims=True)
        running_mean2 = 0.9 * running_mean2 + 0.1 * batch_mean2
        running_var2 = 0.9 * running_var2 + 0.1 * batch_var2
        z2_norm = (z2 - batch_mean2) / np.sqrt(batch_var2 + epsilon)
        batch_normalized_output2 = gamma2 * z2_norm + beta2
        a2 = relu(batch_normalized_output2)

        z3 = np.dot(a2, W3) + b3
        y = softmax(z3)
        
        # loss 계산
        loss = cross_entropy_loss(y, t_batch, weight_decay_lambda)
        train_loss_list.append(loss)

        # backward pass
        dy = (y - t_batch) / x_batch.shape[0]
        dW3 = np.dot(a2.T, dy)
        db3 = np.sum(dy, axis=0, keepdims=True)

        da2 = np.dot(dy, W3.T)
        dz2_relu = da2 * relu_derivative(batch_normalized_output2)
        dgamma2 = np.sum(dz2_relu * z2_norm, axis=0, keepdims=True)
        dbeta2 = np.sum(dz2_relu, axis=0, keepdims=True)
        dz2_norm = dz2_relu * gamma2
        dbatch_var2 = np.sum(dz2_norm * (z2 - batch_mean2) * -0.5 * (batch_var2 + epsilon) ** (-1.5), axis=0, keepdims=True)
        dbatch_mean2 = np.sum(dz2_norm * -1 / np.sqrt(batch_var2 + epsilon), axis=0, keepdims=True) + dbatch_var2 * np.mean(-2 * (z2 - batch_mean2), axis=0, keepdims=True)
        dz2 = dz2_norm / np.sqrt(batch_var2 + epsilon) + dbatch_var2 * 2 * (z2 - batch_mean2) / x_batch.shape[0] + dbatch_mean2 / x_batch.shape[0]

        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(batch_normalized_output)

        dgamma1 = np.sum(dz1 * z1_norm, axis=0, keepdims=True)
        dbeta1 = np.sum(dz1, axis=0, keepdims=True)

        dz1_norm = dz1 * gamma1

        dbatch_var = np.sum(dz1_norm * (z1 - batch_mean) * -0.5 * (batch_var + epsilon) ** (-1.5), axis=0, keepdims=True)
        dbatch_mean = np.sum(dz1_norm * -1 / np.sqrt(batch_var + epsilon), axis=0, keepdims=True) + dbatch_var * np.mean(-2 * (z1 - batch_mean), axis=0, keepdims=True)

        dz1 = dz1_norm / np.sqrt(batch_var + epsilon) + dbatch_var * 2 * (z1 - batch_mean) / x_batch.shape[0] + dbatch_mean / x_batch.shape[0]

        dW1 = np.dot(x_batch.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        update_parameters(dW1, db1, dW2, db2, dW3, db3, learning_rate, weight_decay_lambda)
        gamma1 -= learning_rate * dgamma1
        beta1 -= learning_rate * dbeta1
        gamma2 -= learning_rate * dgamma2
        beta2 -= learning_rate * dbeta2

        

    # epoch 종료 시 정확도 측정
    train_pred = forward_inference(x_train)
    train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(t_train, axis=1))
    train_acc_list.append(train_acc)

    test_pred = forward_inference(x_test)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(t_test, axis=1))
    test_acc_list.append(test_acc)

    print(f"epoch {epoch+1}/{epochs} - loss: {train_loss_list[-1]:.4f} - train acc: {train_acc:.4f} - test acc: {test_acc:.4f}")


# loss 그래프
x_loss = np.arange(len(train_loss_list))
plt.figure(figsize=(10, 4))
plt.plot(x_loss, train_loss_list, label="train loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.show()

# accuracy 그래프
x_acc = np.arange(1, epochs + 1)
plt.figure(figsize=(10, 4))
plt.plot(x_acc, train_acc_list, label="train acc")
plt.plot(x_acc, test_acc_list, label="test acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()
