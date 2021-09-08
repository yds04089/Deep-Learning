import random
import numpy as np
import math
import time

from numpy.core.numeric import base_repr


# overflow를 막기 위해 경우 나눔
def sigmoid_1(z):
    if z < 0:
        return math.exp(z)/(1+math.exp(z))
    else:
        return 1/(1+math.exp(-z))


def sigmoid(z):
    return 1/(1+np.exp(-z))


def loss_func(a, y):
    return -(y*math.log10(a+1e-7)+(1-y)*math.log10(1-a+1e-7))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_sample(n):
    y = []
    x1 = []
    x2 = []
    for i in range(n):
        x1.append(random.uniform(-10, 10))
        x2.append(random.uniform(-10, 10))
        if x1[-1] + x2[-1] > 0:
            y.append(1)
        else:
            y.append(0)
    return x1, x2, y


if __name__ == '__main__':
    alpha = 0.5
    m = 1000
    n = 100
    k = 2000

    # make sample
    x1_train, x2_train, y_train = make_sample(m)
    x1_test, x2_test, y_test = make_sample(n)

    w1 = random.uniform(-10, 10)
    w1_v = w1
    w2 = random.uniform(-10, 10)
    w2_v = w2
    b = random.uniform(-10, 10)
    b_v = b

    # elementwise
    start_element = time.time()
    for idx in range(k):
        J = 0
        dw1 = 0
        dw2 = 0
        db = 0
        accuracy = 0
        for i in range(m):
            z = w1*x1_train[i] + w2*x2_train[i] + b
            a = sigmoid_1(z)
            J += loss_func(a, y_train[i])
            dz = a - y_train[i]
            dw1 += x1_train[i]*dz
            dw2 += x2_train[i]*dz
            db += dz
        J /= m
        dw1 /= m
        dw2 /= m
        db /= m
        w1 = w1 - alpha*dw1
        w2 = w2 - alpha*dw2
        b = b - alpha*db
        # check accuracy every m samples
        '''
        for i in range(m):
            z = w1*x1_train[i]+w2*x2_train[i]+b
            a = sigmoid_1(z)
            if (a > 0.5 and y_train[i] == 1) or (a < 0.5 and y_train[i] == 0):
                accuracy += 1
        print(f"Loss: {J}, Accuracy: {accuracy/m*100}")
        '''
        # 10 iterations 마다 w, b 출력
        if idx % 10 == 9:
            print(f"w1:{w1} w2:{w2} b:{b}")
    element_time = time.time() - start_element

    # vetorization
    start_vectorization = time.time()
    DW = np.zeros((2, 1))
    X1_train = np.array(x1_train)
    X2_train = np.array(x2_train)
    Y_train = np.array(y_train)
    X = np.concatenate([X1_train, X2_train])
    X = np.reshape(X, (2, -1))
    W = np.array([w1_v, w2_v])
    for i in range(k):
        Z = np.dot(W.T, X) + b_v
        A = sigmoid(Z)
        dZ = A - Y_train
        dW = X@dZ.T/m
        dB = np.sum(dZ)/m
        W = W - alpha*dW
        b_v = b_v - alpha*dB

    vectorization_time = time.time() - start_vectorization
    #print(w1, w2, b)
    print("----------------------------------------------------------")
    # test
    # train_samples
    cnt_train = 0
    L = 0
    for i in range(m):
        z = w1*x1_train[i]+w2*x2_train[i]+b
        a = sigmoid_1(z)
        L += loss_func(a, y_train[i])
        if (a > 0.5 and y_train[i] == 1) or (a < 0.5 and y_train[i] == 0):
            cnt_train += 1
    loss_train = L/m
    accuracy_train = (cnt_train/m)*100
    print(f"train samples: accuracy:{accuracy_train}% cost:{loss_train}")

    # test_samples
    cnt_test = 0
    L = 0
    for i in range(n):
        z = w1*x1_test[i]+w2*x2_test[i]+b
        a = sigmoid_1(z)
        L += loss_func(a, y_test[i])
        if (a >= 0.5 and y_test[i] == 1) or (a < 0.5 and y_test[i] == 0):
            cnt_test += 1
    loss_test = L/n
    accuracy_test = (cnt_test/n)*100
    print(f"test samples: accuracy:{accuracy_test}% cost:{loss_test}")
    print(
        f"element_time: {round(element_time, 4)}s | vectorization_time: {round(vectorization_time, 4)}s")
