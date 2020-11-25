import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# importing dataset
(x_train, y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.
x_test = x_test / 255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

'''
model = Sequential([
    Conv2D(32,(3,3),activation='relu',kernel_initializer = 'he_uniform',input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform'),
    Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(100,activation='relu',kernel_initializer='he_uniform'),
    Dense(10,activation='softmax')
    ])

opt = SGD(lr=0.01, momentum=0.9)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,batch_size=32)
model.save('model.h5')

'''

model = load_model('model.h5')
_,acc = model.evaluate(x_test, y_test)
print(acc)
