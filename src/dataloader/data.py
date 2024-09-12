from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

def minst():
    digits = load_digits()

    x, y = digits.images.reshape((len(digits.images), -1)), digits.target


    x_train, x_test, y_train, y_test = train_test_split(x, y);

    return x_train, x_test, y_train, y_test



def cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1) 
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, x_test, y_train, y_test
