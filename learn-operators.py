import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import code


def main():
    MAX_NUM = 2**8-1
    NUM_PAIRS = 60 * 1000
    TRAINING_CUTOFF = 50 * 1000

    sess = tf.InteractiveSession()

    integer_pairs = [
        (random.randint(0, MAX_NUM), random.randint(0, MAX_NUM))
        for _ in range(NUM_PAIRS)
    ]
    greaterthan_answers = np.array([int(b > a) for a, b in integer_pairs])

    integer_bits_pairs = np.array([
        np.concatenate((
            np.unpackbits(np.array([a], dtype=np.uint8)),
            np.unpackbits(np.array([b], dtype=np.uint8))
        ))
        for a, b in integer_pairs
    ])

    training_data = integer_bits_pairs[:TRAINING_CUTOFF]
    test_data = integer_bits_pairs[TRAINING_CUTOFF:]
    training_answers = greaterthan_answers[:TRAINING_CUTOFF]
    test_answers = greaterthan_answers[TRAINING_CUTOFF:]

    # the neural net that solves the "greater than" problem needs only 1 node in
    # the first layer, because the problem of comparing binary numbers is simply
    # taking a linear combination of the bits
    # (1*A[0] + 2*A[1] + ... - 1*B[0] - 2*b[1] - ...) and comparing to 0, but
    # this is precisely what a single node in a neural net can do!
    model = keras.Sequential([
        keras.layers.Dense(1, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(training_data, training_answers, epochs=5)

    test_loss, test_acc = model.evaluate(test_data, test_answers)

    predictions = model.predict(test_data)


if __name__ == '__main__':
    main()
