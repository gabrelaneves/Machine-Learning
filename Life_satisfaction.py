import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Goal: Predict the life satisfaction of a country based on GDP per capita


# Load the data
oecd_bli = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/gdp_per_capita.csv",thousands=',',delimiter='\t',
 encoding='latin1', na_values="n/a")


def prepare_country_stats(oecd_bli,gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli['INEQUALITY']=='TOT']
    oecd_bli = oecd_bli.pivot(index='Country',columns='Indicator',values='Value')
    gdp_per_capita.rename(columns={'2015':'GDP per capita'},inplace=True)
    gdp_per_capita.set_index('Country',inplace=True)
    full_country_stats = pd.merge(left=oecd_bli,right=gdp_per_capita,left_index=True,right_index=True)
    full_country_stats.sort_values(by='GDP per capita',inplace=True)
    remove_indices = [0,1,6,8,33,34,35]
    keep_indices = list(set(range(36))-set(remove_indices))
    return full_country_stats[['GDP per capita', 'Life satisfaction']].iloc[keep_indices]

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)


X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#--------Splitting the data--------
X_train, X_test, Y_train ,Y_test, = train_test_split(X, y , test_size=0.3, train_size=0.7, random_state=42)

#--------Visualize the training and test data

plt.plot(X_train, Y_train, 'bo', label ="Train")
plt.plot(X_test, Y_test, 'bo', color="r", label ="Test")
plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.legend(loc="upper left")
plt.title("Dataset")
plt.show()


#-----Selecting the linear model
model = sklearn.linear_model.LinearRegression()
#----Train the model
model.fit(X_train, Y_train)


#----predicting Y test values
Y_predic = model.predict(X_test)

#----calculating the distance between the real values and 
#the predict values to evaluate the model efficiency


error = mean_squared_error(Y_test, Y_predic)

#----return 0.13, which is a good value that indicates that the
#real values are not too distance from the predict values
#but may also indicates overfitting


#ploting the predict vs real values
plt.plot(X_test, Y_test, 'bo', label = "Test")

plt.plot(X_test, Y_predic, 'bo', color="r", label = "Predict")

plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.legend(loc="upper left")
plt.title("Predict vs Real values")
plt.show()

#----------------- SECTION 2------------
#genarating random numbers and predicting the Life satisfaction

x_new = [randint(10000,70000) for p in range(0, 30)]
x_new = np.array(x_new)
lista = []

for x in x_new:
    x = [[x]]
    y = model.predict(x)
    lista.append(y[0][0])

plt.plot(x_new.tolist(),lista,  'bo')
plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.title("Random values")
plt.show()