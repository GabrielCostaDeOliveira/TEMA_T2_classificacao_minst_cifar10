import pandas as pd
from sklearn.datasets import load_digits

def minst():
    digits = load_digits()

    return digits.images.reshape((len(digits.images), -1)), digits.target
