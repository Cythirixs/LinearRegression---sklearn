import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('challenge_dataset.txt', names = ['X', 'Y'])

X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(df['X'], df['Y'], test_size = 0.1))


reg = LinearRegression()
reg.fit(X_train.values.reshape(-1, 1), Y_train.values.reshape(-1, 1))


print('Score: ', reg.score(X_train.values.reshape(-1, 1), Y_train.values.reshape(-1, 1)))

sns.regplot(x = 'X', y = 'Y', data = df, fit_reg = False)
plt.plot(df['X'].values.reshape(-1, 1), reg.predict(df['X'].values.reshape(-1, 1)))
plt.show()
