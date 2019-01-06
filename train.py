import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import neural_net

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#show sample train data 
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

model = neural_net.neural_net()
model.fit(x_train, y_train, epochs=2)

val_loss, val_acc = model.evaluate(x_test, y_test)
print('loss: ' + val_loss, 'accuracy: ' + val_acc)

model.save_weights('model.h5')
print('Model has been saved')

#save model
#model_json = model.to_json()
#with open('model.json', 'w') as json_file:
#    json_file.write(model_json)
#model.save_weights('model.h5')
#print('Model has been saved')

#model.save('mmm.h5')

#test
#print(y_train[0])
#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()
