import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import seaborn as sns


# rng = np.random.default_rng(123)
# data = np.array([[4, 15], [6, 8], [5, 17], [8, 15], [3, 1]])

# x = np.arange(0, 10, 0.1)
# y = 2 * x  + 1

# plt.plot(x, y)
# plt.scatter(data[:, 0], data[:, 1])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.xlim(0, 10)
# plt.yticks([0, 5, 10, 15, 20])
# plt.show()


data = np.array([[4, 15], [6, 8], [5, 17], [8, 15], [3, 1]])


#ここでパラメータ変更
a = 2
b = 1

x = np.arange(0, 10, 0.1)
y1 = a * x + b
y2 = np.mean(data[:, 1])

print(y2)
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'cm'

# ax.plot(x, y)
# ax.scatter(data[:, 0], data[:, 1], color='red')
# for i, x in enumerate(data[:, 0]):
#     y_hat = a * x + b
    # if max(y_hat, data[i, 1]) == data[i, 1]:
    #     ax.text(data[i, 0], data[i, 1], f'$(x_{i+1}, y_{i+1})$', ha='center', va='bottom', size=15)
    # else:
    #     ax.text(data[i, 0], data[i, 1], f'$(x_{i+1}, y_{i+1})$', ha='center', va='top', size=15)
    # ax.vlines([x], ymin=min(y_hat, data[i, 1]), ymax=max(y_hat, data[i, 1]), color='green', linestyles='dashed')
    # ax.text(data[i, 0], (data[i, 1] + y_hat) / 2, f'$\epsilon_{i+1}$', ha='left', size=15)
# ax.set_xlabel(r'$x$', fontsize=15)
# ax.set_ylabel(r'$y$', fontsize=15)

# plt.show()


# dataset = fetch_california_housing(as_frame=True)
# df = dataset.frame
# df['Price'] = dataset.target
# print(dataset.DESCR)

# diabetes = datasets.load_diabetes()

# df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# X = pd.concat([pd.Series(np.ones(len(df['bmi']))), df.loc[:, ['bmi', 's5']]], axis=1, ignore_index=True).values
# y = diabetes.target

# print(X)
