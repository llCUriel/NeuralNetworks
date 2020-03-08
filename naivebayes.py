import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, classification_report
sns.set()
sns.set_palette("Dark2")

N_pts = 1000
centers = 3
X, Y = make_blobs(N_pts, centers=centers)


sns.scatterplot(X[:, 0], X[:, 1], hue=Y, palette="Dark2")

"""### Construir el modelo"""

class BayesClassifier():
    """Clasificador naive de Bayes, se asume que los datos están distribuidos
    de forma normal y que todas las clases son igual de probables""" 

    def __init__(self):
        self.mu = []
        self.sd = []
        self.N = 0
        self.centers = 0

    def fit(self, X, Y):
        """ Se encuentran los parámetros de cada distribución,
        asumiendo que están distriuidas de forma normal """
        self.N = X.shape[1]
        self.centers = max(np.unique(Y)) + 1
        self.mu = np.zeros((self.centers, self.N))
        self.sd = np.zeros((self.centers, self.N))
        

        for y in range(self.centers):
            for n in range(self.N):
                mu_i = np.mean(X[Y==y, n])
                sd_i = np.std(X[Y==y, n])

                self.mu[y, n] = mu_i
                self.sd[y, n] = sd_i

    def predict(self, X):
        P = np.ones((len(X), self.centers))

        for y in range(self.centers):
            P_aux = np.ones(len(X))
            for n in range(self.N):
                P_aux = P_aux * norm.pdf(X[:, n],
                                         loc=self.mu[y, n],
                                         scale=self.sd[y, n])
            P[:, y] = P_aux
        return P

# Inicializar el clasificador y entrenarlo
bayes = BayesClassifier()

bayes.fit(X, Y)

# Se pueden revisar los parámetros que ha ajustado del modelo 
print("Medias:\n", bayes.mu)

print("Desviación estándar:\n", bayes.sd)

"""### Resultados del clasificador"""

# Obtener el resultado de la clasificación
y_hat = np.argmax(bayes.predict(X), 1)

# Revisar los resultados
print(confusion_matrix(Y, y_hat))

print(classification_report(Y, y_hat))

"""### Visualizar la superficie de separación"""

# Paso del grid
h = 0.1

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

Z = np.argmax(bayes.predict(np.c_[xx.ravel(), yy.ravel()]), 1)
Z = Z.reshape(xx.shape)

cm = plt.cm.viridis
sns.scatterplot(X[:, 0], X[:, 1], hue=Y, palette=cm)
plt.contourf(xx, yy, Z, alpha=.3, cmap=cm)
plt.show()

np.randint(19)+1

