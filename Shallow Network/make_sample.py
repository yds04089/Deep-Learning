import random
import csv


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
    m = 10000
    n = 500
    k = 5000

    # make sample
    x1_train, x2_train, y_train = make_sample(m)
    x1_test, x2_test, y_test = make_sample(n)
    with open('sample.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(x1_train)
        writer.writerow(x2_train)
        writer.writerow(y_train)
        writer.writerow(x1_test)
        writer.writerow(x2_test)
        writer.writerow(y_test)
