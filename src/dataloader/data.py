import pandas as pd
from sklearn.datasets import load_digits

def minst() -> pd.DataFrame:
    digits = load_digits()

    # Correcting the syntax and formatting
    return pd.DataFrame({
        'x': list(digits.images.reshape((len(digits.images), -1))/255),  # Added the missing parenthesis and comma
        'y': digits.target
    })
