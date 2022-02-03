import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import pandas as pd
import seaborn as sns

import code


dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
)

column_names = [
    'MPG',
    'Cylinders',
    'Displacement',
    'Horsepower',
    'Weight',
    'Acceleration',
    'Model Year',
    'Origin',
]
raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment='\t',
    sep=" ",
    skipinitialspace=True,
)

dataset = raw_dataset.copy()
dataset.tail()

dataset.dropna()
## TODO: Horsepower has some nan's which might be leading to messed up results

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

train_stats = train_dataset.describe().transpose()

def norm(data):
    return (data - train_stats['mean']) / train_stats['std']

norm_train_dataset = norm(train_dataset)
norm_test_dataset = norm(test_dataset)

def simple_model():
    model = keras.Sequential([
        layers.Dense(
            64,
            activation=tf.nn.relu,
            input_shape=[len(train_dataset.keys())],
        ),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1),
    ])
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.RMSprop(0.001),
        metrics=['mean_absolute_error', 'mean_squared_error'],
    )
    return model

model = simple_model()

history = model.fit(
    norm_train_dataset,
    train_labels,
    epochs=1000,
    validation_split=0.2,
)



code.interact(local=locals())
