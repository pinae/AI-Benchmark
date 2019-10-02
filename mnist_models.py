# -*- coding: utf-8 -*-
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_dataset(batch_size=128):
    train_data, test_data = tf.keras.datasets.mnist.load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data
    img_size = x_train.shape[1:3]
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    if tf.keras.backend.image_data_format() == 'channels_last':
        x_train = x_train.reshape(train_size, img_size[0], img_size[1], 1)
        x_test = x_test.reshape(test_size, img_size[0], img_size[1], 1)
        input_shape = (img_size[0], img_size[1], 1)
    else:
        x_train = x_train.reshape(train_size, 1, img_size[0], img_size[1])
        x_test = x_test.reshape(test_size, 1, img_size[0], img_size[1])
        input_shape = (1, img_size[0], img_size[1])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (
        input_shape,
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size))


def flat_model(input_shape=(28, 28, 1), num_classes=10):
    seq_model = tf.keras.Sequential()
    seq_model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    seq_model.add(tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax))
    return seq_model


if __name__ == "__main__":
    in_shape, train_dataset, test_dataset = get_dataset(batch_size=200)
    model = flat_model(in_shape, 10)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.categorical_accuracy])
    history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)
    print('\n# Evaluate')
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
