import tensorflow as tf
import random
import numpy as np
import time


def make_sample(n):
    y = []
    x = []
    for i in range(n):
        x1 = random.uniform(-10, 10)
        x2 = random.uniform(-10, 10)
        x.append([x1, x2])
        if x1 + x2 > 0:
            y.append(1)
        else:
            y.append(0)
    return np.array(x), np.array(y)


if __name__ == '__main__':
    m = 10000
    n = 500

    # make sample
    x_train, y_train = make_sample(m)
    x_test, y_test = make_sample(n)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    model.fit(x_train, y_train, batch_size=128, epochs=1000, verbose=1)
    end = time.time()

    model.evaluate(x_train,  y_train, verbose=2)
    model.evaluate(x_test,  y_test, verbose=2)
    print(round(end - start, 4))
