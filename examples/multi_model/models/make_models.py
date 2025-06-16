import pathlib

import fire
import sonnet as snt
import tensorflow as tf

class Model(snt.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.linear = snt.Linear(num_channels)

    @tf.function(input_signature=[tf.TensorSpec([None, 10])])
    def __call__(self, x):
        return self.linear(x)

def main():
    model_1 = Model(1)
    model_2 = Model(2)

    x = tf.random.normal((10, 10))
    y = model_1(x)

    print(y)
    print(model_2(x))
    print(model_1.trainable_variables)

    tf.saved_model.save(model_1, str(pathlib.Path(__file__).parent / 'model_1'))
    tf.saved_model.save(model_2, str(pathlib.Path(__file__).parent / 'model_2'))

    model_3 = tf.saved_model.load(str(pathlib.Path(__file__).parent / 'model_1'))
    print(model_3.__call__(x))

if __name__ == '__main__':
    fire.Fire(main)
