### o objetivo final é criar um modelo que decida qual é a melhor droga\
# a ser prescrita com base nas caracteristicas de tratamento de outros pacientes

##Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier##importing librarys

#carregando base de dados
df = pd.read_csv(r"C:\Users\Gabriela Neves\Downloads\drug200.csv", delimiter=",")

 #separando variaveis independentes e dependentes e convertendo em um array

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

#Como no dataset existem atributos de classificação, é necessário converter essas features para classificações numericas
##para isso, uso Preporocessing, o metodo de tratar dados do sklearn
from sklearn import preprocessing

encd_1 = preprocessing.LabelEncoder()
###passo os tipos de classificação que existem nessa coluna para serem convertidos
##no caso dessa coluna se tem a classificação F e M
encd_1.fit(['F','M'])

#Transformando a primeira coluna do array
X[:,1] = encd_1.transform(X[:,1])

##repetindo o processo para as outras colunas classificatórias
##BP
encd_2 = preprocessing.LabelEncoder()
encd_2.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = encd_2.transform(X[:,2])

###Cholesterol
encd_3 = preprocessing.LabelEncoder()
encd_3.fit(['NORMAL', 'HIGH'])
X[:,3] = encd_3.transform(X[:,3])
X[0:5]

###Definindo variavel de saida
Y = df['Drug']

###separando data frame em partes de treino (70%) e teste (30%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 0.3, random_state =3)

####Criando arvore
from sklearn.tree import DecisionTreeClassifier
#atribuindo metodo a uma variavel
#parametros = quao profunda a arvore sera e qual o metodo que ela usara para escolher as melhores features para separa
##nesse caso, usarei o nivel de entropia do sistema

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
##treiando modelo
tree.fit(X_train,Y_train)

##prevendo
y_hat = tree.predict(X_test)
Y_a = Y_test.values
##comparando Y predict com o Y real
print('Predict Result:', y_hat[0:7])
print('Real Result:', Y_a[0:7])

##verificando a eficiencia do modelo baseado em sua acuracia
from sklearn import metrics
print('Precisão do modelo:', metrics.accuracy_score(Y_test, y_hat))

###O modelo apresentou cerca de 98% de precisão
