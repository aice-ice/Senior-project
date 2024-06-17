import numpy as np
from scipy import linalg
import csv
import urllib.request
import io
import zipfile
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.w_ = None
    
    def fit(self, X, t):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        A = np.dot(Xtil.T, Xtil)
        b = np.dot(Xtil.T, t)
        self.w_ = linalg.solve(A, b)
        
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)
    

url = 'https://archive.ics.uci.edu/static/public/186/wine+quality.zip'

req = urllib.request.Request(url)
with urllib.request.urlopen(req) as res:
    data = res.read()

with zipfile.ZipFile(io.BytesIO(data), 'r') as zip_data:
    csv_filepath = [name for name in zip_data.namelist() if '.csv' in name][0]
    with zip_data.open(csv_filepath, 'r') as csv:
        df = csv.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(df), sep=';')
        
Xy = df.to_numpy()

np.random.seed(0)
np.random.shuffle(Xy)

train_X = Xy[:-1000, :-1]
train_y = Xy[:-1000, -1]
test_X = Xy[-1000:, :-1]
test_y = Xy[-1000:, -1]

model = LinearRegression()
model.fit(train_X, train_y)

y = model.predict(test_X)

print('最初の５つの正解と予測値')
for i in range(5):
    print(f'{test_y[i]} {y[i]}')
print()
print('RMSE:', np.sqrt(((test_y - y)**2).mean()))

