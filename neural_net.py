import tensorflow as tf

def neural_net(load=''):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape(784,)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(  optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    verbose=2)

    if load:
        model.load_weights(load)
        
    model.summary()

    return model
