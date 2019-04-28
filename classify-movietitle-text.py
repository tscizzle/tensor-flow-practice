import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import code


def main():
    imdb = keras.datasets.imdb

    VOCAB_SIZE = 10 * 1000

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=VOCAB_SIZE
    )

    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = {num: word for word, num in word_index.items()}

    def view_text(datum):
        return ' '.join(reverse_word_index.get(num, '*') for num in datum)

    train_data = keras.preprocessing.sequence.pad_sequences(
        train_data,
        value=word_index["<PAD>"],
        padding='post',
        maxlen=256
    )

    test_data = keras.preprocessing.sequence.pad_sequences(
        test_data,
        value=word_index["<PAD>"],
        padding='post',
        maxlen=256
    )

    model = keras.Sequential()
    model.add(keras.layers.Embedding(VOCAB_SIZE, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc']
    )

    partial_train_data = train_data[10000:]
    partial_train_labels = train_labels[10000:]

    val_data = train_data[:10000]
    val_labels = train_labels[:10000]

    history = model.fit(
        partial_train_data,
        partial_train_labels,
        epochs=40,
        batch_size=512,
        validation_data=(val_data, val_labels),
        verbose=True,
    )

    results = model.evaluate(test_data, test_labels)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    def predict(sentence):
        # try out different sentences with predict('i liked this movie')
        words = sentence.split(' ')
        word_ints = [word_index[word] for word in words]
        return model.predict(np.array([word_ints]))

    code.interact(local=locals())


if __name__ == '__main__':
    main()
