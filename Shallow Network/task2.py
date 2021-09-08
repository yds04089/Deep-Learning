import random
import numpy as np
import math
import time
import csv
from numpy.core.numeric import base_repr


# overflow를 막기 위해 경우 나눔
def sigmoid_1(z):
    if z < 0:
        return math.exp(z)/(1+math.exp(z))
    else:
        return 1/(1+math.exp(-z))


def loss_func(a, y):
    return -(y*math.log10(a+1e-7)+(1-y)*math.log10(1-a+1e-7))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_(z):
    return sigmoid(z)*(1-sigmoid(z))


def get_sample():
    x1_train = []
    x2_train = []
    y_train = []
    x1_test = []
    x2_test = []
    y_test = []
    with open('sample.csv', 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            if i == 0:
                for x in line:
                    x1_train.append(float(x))
            elif i == 1:
                for x in line:
                    x2_train.append(float(x))
            elif i == 2:
                for x in line:
                    y_train.append(int(x))
            elif i == 3:
                for x in line:
                    x1_test.append(float(x))
            elif i == 4:
                for x in line:
                    x2_test.append(float(x))
            elif i == 5:
                for x in line:
                    y_test.append(int(x))
    return x1_train, x2_train, y_train, x1_test, x2_test, y_test


if __name__ == '__main__':
    alpha = 0.1
    m = 10000
    n = 500
    k = 5000

    # make sample
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = get_sample()

    w1_1 = random.uniform(-10, 10)
    w1_2 = random.uniform(-10, 10)
    w2 = random.uniform(-10, 10)
    b1 = random.uniform(-10, 10)
    b2 = random.uniform(-10, 10)

    # vetorization
    start_vectorization = time.time()
    DW = np.zeros((2, 1))
    X1_train = np.array(x1_train)
    X2_train = np.array(x2_train)
    Y_train = np.array(y_train)
    X = np.concatenate([X1_train, X2_train])
    X = np.reshape(X, (2, -1))
    W1 = np.array([w1_1, w1_2])
    W2 = np.array(w2)
    for i in range(k):
        # forward propagation
        Z1 = np.dot(W1.T, X) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2.T, A1)+b2
        A2 = sigmoid(Z2)
        # back propagation
        dZ2 = A2 - Y_train
        dW2 = dZ2@A1.T/m
        dB2 = np.sum(dZ2)/m
        dZ1 = W2*dZ2 * sigmoid_(Z1)
        dW1 = (dZ1@X.T)/m
        dB1 = (np.sum(dZ1))/m
        W1 = W1 - alpha*dW1
        b1 = b1 - alpha*dB1
        W2 = W2 - alpha*dW2
        b2 = b2 - alpha*dB2
        # 50 iterations 마다 w, b 출력
        if i % 50 == 49:
            print(f"w1:{W1[0]} {W1[1]} b1:{b1} w2:{W2}  b2:{b2}")
    vectorization_time = time.time() - start_vectorization
    #print(w1, w2, b)
    print("----------------------------------------------------------")
    w1_1 = W1[0]
    w1_2 = W1[1]
    w2 = W2
    # test
    # train_samples
    start_test = time.time()
    cnt_train = 0
    L = 0
    for i in range(m):
        z1 = w1_1*x1_train[i]+w1_2*x2_train[i]+b1
        a1 = sigmoid_1(z1)
        z2 = w2*a1+b2
        a2 = sigmoid_1(z2)
        L += loss_func(a2, y_train[i])
        if (a2 > 0.5 and y_train[i] == 1) or (a2 < 0.5 and y_train[i] == 0):
            cnt_train += 1
    loss_train = L/m
    accuracy_train = (cnt_train/m)*100
    print(f"train samples: accuracy:{accuracy_train}% cost:{loss_train}")

    # test_samples
    cnt_test = 0
    L = 0

    for i in range(n):
        z1 = w1_1*x1_test[i]+w1_2*x2_test[i]+b1
        a1 = sigmoid_1(z1)
        z2 = w2*a1+b2
        a2 = sigmoid_1(z2)
        L += loss_func(a2, y_test[i])
        if (a2 >= 0.5 and y_test[i] == 1) or (a2 < 0.5 and y_test[i] == 0):
            cnt_test += 1
    test_time = time.time() - start_test
    loss_test = L/n
    accuracy_test = (cnt_test/n)*100
    print(f"test samples: accuracy:{accuracy_test}% cost:{loss_test}")
    print(
        f"train_time: {round(vectorization_time, 4)}s")
    print(
        f"test_time: {round(test_time, 4)}s")
