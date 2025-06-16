import pathlib

import fire
import tensorflow as tf

def main():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    x = tf.random.normal((10, 10))
    model(x)

    tf.saved_model.save(model, str(pathlib.Path(__file__).parent / 'model'))


if __name__ == '__main__':
    fire.Fire(main)
