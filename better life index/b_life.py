import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.neighbors
# ................... #

oecd_bli = pd.read_csv('oecd_bli_2015.csv', thousands=',')
gdp_per_capita = pd.read_csv('gdp_per_capita.csv', encoding='latin1', na_values='n/a')
oecd_bli = oecd_bli[(oecd_bli['Inequality'] == 'Total') &
                    (oecd_bli['Indicator'] == 'Life expectancy')]

# ................... #

combined_data = pd.merge(gdp_per_capita, oecd_bli, on=['Country'])

gdp_value = combined_data[['2015']].copy()
bli_value = combined_data[['Value']].copy()

gdp_value.columns = ['GDP per capita']
bli_value.columns = ['Life satisfaction']

country_stats = pd.concat([gdp_value, bli_value], axis=1)

# ................... #

X = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]


country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
#plt.show()

# ................... #

model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)
X_new = [[22587]]


print (model.predict(X_new))