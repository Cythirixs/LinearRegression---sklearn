import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Read
df = pd.read_excel('anime stats.xlsx', sheetname= 'Sheet1')

#Find Error
yr_train, yr_test, profit_train, profit_test = np.asarray(train_test_split(df['Year'], df['Total Profit'], test_size = 0.1))

reg = LinearRegression()
reg.fit(yr_train.values.reshape(-1, 1), profit_train.values.reshape(-1, 1))

print('Score : ', reg.score(yr_test.values.reshape(-1, 1), profit_test.values.reshape(-1, 1)))

#Visualize
sns.regplot(x = 'Year', y = 'Total Profit', data = df, fit_reg= False)

reg_line = LinearRegression(normalize=True)
reg_line.fit(df['Year'].values.reshape(-1, 1), df['Total Profit'].values.reshape(-1, 1))

plt.plot(df['Year'].values.reshape(-1, 1), reg_line.predict(df['Year'].values.reshape(-1, 1)))
plt.show()



