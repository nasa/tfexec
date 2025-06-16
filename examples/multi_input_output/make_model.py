import fire
import sonnet as snt
import tensorflow as tf

class Model(snt.Module):
    def __init__(self, num_channels_1, num_channels_2):
        super().__init__()
        self.linear_1 = snt.Linear(num_channels_1)
        self.linear_2 = snt.Linear(num_channels_2)

    @tf.function(input_signature=[tf.TensorSpec([None, 5]), tf.TensorSpec([None, 4])])
    def __call__(self, x_1, x_2):
        x = tf.concat([x_1, x_2], axis=-1)
        return self.linear_1(x), self.linear_2(x)

def main():
    model = Model(1, 2)

    x_1 = tf.random.normal((10, 5))
    x_2 = tf.random.normal((10, 4))
    print(model(x_1, x_2))
    
    tf.saved_model.save(model, 'model')

if __name__ == '__main__':
    fire.Fire(main)