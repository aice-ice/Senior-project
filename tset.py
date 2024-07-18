# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# import pandas as pd
# import seaborn as sns
# import urllib.request

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


# data = np.array([[4, 15], [6, 8], [5, 17], [8, 15], [3, 1]])


# #ここでパラメータ変更
# a = 2
# b = 1

# x = np.arange(0, 10, 0.1)
# y1 = a * x + b
# y2 = np.mean(data[:, 1])

# # print(y2)
# # fig = plt.figure(figsize=(8, 5))
# # ax = fig.add_subplot(111)

# # plt.rcParams['font.family'] = 'Times New Roman'
# # plt.rcParams['mathtext.fontset'] = 'cm'

# # ax.plot(x, y)
# # ax.scatter(data[:, 0], data[:, 1], color='red')
# # for i, x in enumerate(data[:, 0]):
# #     y_hat = a * x + b
#     # if max(y_hat, data[i, 1]) == data[i, 1]:
#     #     ax.text(data[i, 0], data[i, 1], f'$(x_{i+1}, y_{i+1})$', ha='center', va='bottom', size=15)
#     # else:
#     #     ax.text(data[i, 0], data[i, 1], f'$(x_{i+1}, y_{i+1})$', ha='center', va='top', size=15)
#     # ax.vlines([x], ymin=min(y_hat, data[i, 1]), ymax=max(y_hat, data[i, 1]), color='green', linestyles='dashed')
#     # ax.text(data[i, 0], (data[i, 1] + y_hat) / 2, f'$\epsilon_{i+1}$', ha='left', size=15)
# # ax.set_xlabel(r'$x$', fontsize=15)
# # ax.set_ylabel(r'$y$', fontsize=15)

# # plt.show()


# # dataset = fetch_california_housing(as_frame=True)
# # df = dataset.frame
# # df['Price'] = dataset.target
# # print(dataset.DESCR)

# # diabetes = datasets.load_diabetes()

# # df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# # X = pd.concat([pd.Series(np.ones(len(df['bmi']))), df.loc[:, ['bmi', 's5']]], axis=1, ignore_index=True).values
# # y = diabetes.target

# # print(X)

# # url = 'https://book.mynavi.jp/support/e2/9784839965259/df_samplefiles.zip'
# # path = R"C:\Users\eu21052\Desktop\Python\df_samplefiles.zip"

# # urllib.request.urlretrieve(url, path)

# data = np.array(np.sin(np.radians([20, 80, 110, 190, 210])))

# x = np.array([[data[i] ** j for j in range(8)] for i in range(len(data))])

# print(np.linalg.det(x.T @ x))

import numpy as np
import random as rnd
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def poly(intercept, coef, x):
    return intercept + sum([w * x**(n + 1) for n, w in enumerate(coef)])

rnd.seed(0)
np.random.seed(0)
xmin, xmax = -5, 5
xlim_min, xlim_max = -20, 20
ylim_min, ylim_max = -1, 30

n_data = 9
n_features = 20
n_terms_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

x = np.linspace(xmin, xmax, n_data)
y = np.array(x ** 2 + np.random.normal(0, 1, n_data))

df = pd.DataFrame(y, columns=['y'])
for n in range(n_features):
    df["x^{}".format(n+1)] = x**(n+1)
# print(df)

fig, axs = plt.subplots(3, 3, figsize=(12, 6.4))
axs_1d = axs.reshape(1, -1)[0]

linreg = LinearRegression()

x_graph = np.linspace(xlim_min, xlim_max)
w_df = pd.DataFrame()
for ax, n_terms in zip(axs_1d, n_terms_list):
    linreg.fit(df.iloc[:, 1:n_terms+1], df['y'])
    y_linreg = poly(linreg.intercept_, linreg.coef_, x_graph)
    ax.scatter(df['x^1'], df['y'], c='r', zorder=10)
    ax.plot(x_graph, y_linreg, c='gray', linewidth=2,
        label="n_terms={}".format(n_terms))
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')

    print(f'w_{n_terms} : {np.linalg.norm(linreg.coef_)}')

plt.show()

