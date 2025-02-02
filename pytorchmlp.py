import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import torch
from torch import optim
from torch import nn
import seaborn as sns
import imageio

# Inicializar las propiedades de seaborn
sns.set()
sns.set_palette("Dark2")

# Función para crear los GIF
def CrearGIF(images, fname="aprendizaje"):
    with imageio.get_writer(fname + ".gif", mode="I") as writer:
        for image in images:
            im = imageio.imread(image)
            writer.append_data(im)

# Función para crear datos del problema XOR
def make_XOR(N_pts, noise=0):
    X1 = np.random.randn(int(N_pts/4), 2) * noise + [0.0]
    X2 = np.random.randn(int(N_pts/4), 2) * noise + [3, 3]
    X3 = np.random.randn(int(N_pts/4), 2) * noise + [0, 3]
    X4 = np.random.randn(int(N_pts/4), 2) * noise + [3, 0]

    X = np.r_[X1, X2, X3, X4]
    Y = np.r_[np.ones(int(N_pts/2)).T, np.zeros(int(N_pts/2)).T]
    return X, Y

"""## Datos a usar

En este caso se crean conjuntos de datos artificiales no linealmente separables para el entrenamiento de las redes.
"""

N_pts = 1000

# Usar datos del problema XOR
#X, Y = make_XOR(N_pts, noise=0.3)

# Usar datos de medias lunas
#X, Y = make_moons(N_pts, noise=0.05)

# Usar datos de circulos 
X, Y = make_circles(N_pts, noise=0.05, factor=0.6)


sns.scatterplot(X[:, 0], X[:, 1], hue=Y)

X_t = torch.from_numpy(X).float()
Y_t = torch.from_numpy(Y).float()

"""## Definir una red neuronal

En esta primera parte se define una red neuronal usando las funciones de álgebra lineal de Pytorch. Se propone una red de 2 capas ocultas y una
"""



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Definir la activación intermedia
        self.intermedia = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 2),
            nn.Tanh(),
        )

        # Definir la activación final
        self.salida = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()   
        )

    def forward(self, x):
        # Salida de las etapas intermedias
        x = self.intermedia(x)
        
        # Salida de la etapa final
        x = self.salida(x)
        
        return x

# Definir la red, en este caso no tiene parametros de entrada
net = MLP()

# Definir el optimizador
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# Definir la función de costo a optimizar 
loss_func = torch.nn.BCELoss()

images = []
for t in range(100):
    out = net(X_t)                 
    loss = loss_func(out, Y_t)
    inter = net.intermedia(X_t).data.numpy()
    sns.scatterplot(inter[:, 0], inter[:, 1], hue=Y)
    plt.axis("off")
    plt.savefig("fig" + str(t+1).zfill(2)+".png")
    plt.clf()
    images.append("fig" + str(t+1).zfill(2)+".png")

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()

out_np = np.squeeze(out.data.numpy())
sns.scatterplot(X[:, 0], X[:, 1], hue=(out_np>0.5)*1)

CrearGIF(images)

"""## Definir elementos de la red

Pytorch permite crear tus propias funciones de activación, de costo e incluso capas para una red.

En general esto se realiza creando una clase nueva con herencia de ```nn.Module```

### Definir función de activación


En este caso se define la función de activación _Swish_:
$$Swish(x) = x * \beta \sigma(x)$$

Con $\sigma$ siendo la función sigmoide y $\beta$ una constante. Fue propuesta inicialmente en: SWISH: A SELF-GATED ACTIVATION FUNCTION https://arxiv.org/pdf/1710.05941v1.pdf
"""

# Definir función de activación
def swish(x):
    return x * torch.sigmoid(x)

# Definir la clase a usar
class Swish(nn.Module):
    def __init__(self):
        '''
        Inicialización de la clase, se le pueden dar parametros
        '''
        super().__init__()

    def forward(self, input):
        '''
        Evaluar hacia adelante
        '''
        return swish(input)

# Evaluar la función
X_swish = torch.arange(-5, 5, 0.1)
Y_swish = swish(X_swish)
sns.lineplot(X_swish, Y_swish)
plt.title("Swish: $x * \sigma(x)$")
plt.show()

class MLPSwish(nn.Module):
    def __init__(self):
        super().__init__()
        
        #self.softmax = nn.Softmax(dim=1)

        self.intermedia = nn.Sequential(
            nn.Linear(2, 10),
            Swish(),
            nn.Linear(10, 2),
            nn.Tanh(),
        )

        self.salida = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()   
        )

    def forward(self, x):
        # Salida de las etapas intermedias
        x = self.intermedia(x)
        
        # Salida de la etapa final
        x = self.salida(x)
        
        return x

# Definir la red, en este caso no tiene parametros de entrada
netSwish = MLPSwish()

# Definir el optimizador
optimizer = torch.optim.Adam(netSwish.parameters(), lr=0.05)

# Definir la función de costo a optimizar 
loss_func = torch.nn.BCELoss()

images_swish = []
for t in range(100):
    out = netSwish(X_t)                 
    loss = loss_func(out, Y_t)
    inter = netSwish.intermedia(X_t).data.numpy()
    sns.scatterplot(inter[:, 0], inter[:, 1], hue=Y)
    plt.axis("off")
    plt.savefig("fig" + str(t+1).zfill(2)+".png")
    plt.clf()
    images_swish.append("fig" + str(t+1).zfill(2)+".png")

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()

out_np = np.squeeze(out.data.numpy())
sns.scatterplot(X[:, 0], X[:, 1], hue=(out_np>0.5)*1)

CrearGIF(images_swish, "apr_swish")

