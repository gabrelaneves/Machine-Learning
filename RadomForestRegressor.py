#------------------------- L I B R A R Y S ------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# ------------  LOADING DATA  ----------------
df = pd.read_csv("advertising.csv")

df.info()

sns.heatmap(df.corr(), cmap = 'Wistia', annot =True)
plt.show()


#definindo variaveis independentes e dependentes

x = df.drop("Vendas", axis = 1)
y = df["Vendas"]

#definindo variaveis de treino e teste

x_treino,x_teste,y_treino,y_teste = train_test_split(x, y, test_size = 0.3) 

# maior parte dos dados é direcionada para o aprendizado da maquina, 30% são para teste, 70% para ap

#Usando dois metodos para ver qual trará melhores resultados


# ------------  INITIALIZING MODEL  ----------------

linear = LinearRegression()
randomforest = RandomForestRegressor()

# Treinando pelo sistema linear e random
linear.fit(x_treino, y_treino)
randomforest.fit(x_treino, y_treino)

#Testando qual funcionou melhor, usando metrics

#  partir dos valores de x existentes ele preverá um valor para y

teste_linear = linear.predict(x_teste)

teste_random = randomforest.predict(x_teste)

#  comparando se os valores de Y existentes são parecido com os que ele preveu anteriormente
#Metrica r2, quantos por cento de precisão com os valores originais

r2_linear = metrics.r2_score(y_teste, teste_linear)
r2_random = metrics.r2_score(y_teste, teste_random)
print(r2_linear, r2_random)

#taxa de incerteza

m_linear = metrics.mean_squared_error(y_teste, teste_linear)

m_random = metrics.mean_squared_error(y_teste, teste_random)

print(m_linear, m_random)

# A partir dos ultimos resultados, percebemos que o metodo de aprendizagem rangem foi maos eficaz

#criando tabela para comparar
analise = pd.DataFrame()
analise["Vendais reais"] = y_teste
analise['Previsão'] =  teste_random
analise = analise.reset_index(drop = True)

sns.lineplot(data = analise)
plt.show()
