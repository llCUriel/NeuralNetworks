import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

"""## Estimar una distribución de Bernoulli"""

# Parámetro para Bernoulli
p = 0.7

N = 1000

# Generar las muestras
D = np.array([np.random.random() < p for _ in range(N)]) * 1.0
print("Parametro real:", p)

# Histograma
plt.figure(dpi=200)
plt.title("Distribución de la muestra")
plt.bar([0, 1], [np.sum(D==0), np.sum(D==1)])
plt.show()

# Pasar los datos a un tensor de pytorch
D_t = torch.from_numpy(D).float()

# Parámetro estimado a optimizar
p_hat = torch.rand((1,1), requires_grad=True)
print("Parametro inicial:", p_hat.item())

# Ciclo de optimizacion por descenso de gradiente
iter = 200
alpha = 0.01

# Guardar el costo en cada paso
c = []

for _ in range(iter):
    # Calcular la función de costo
    l = -torch.mean(D_t * torch.log(p_hat) + (1-D_t) * torch.log(1 - p_hat))

    c.append(l.item())

    # Estimar gradiente de las variables dependientes
    l.backward()

    # Actualizar el valor (se apaga el gradiente)
    with torch.no_grad():
        p_hat -= alpha * p_hat.grad

        p_hat.grad.zero_()

plt.figure(dpi=200)
plt.plot(c)
plt.title("Verosimilitud logarítmica negativa por época")
plt.xlabel("Iteración")
plt.ylabel("$-log L(D|\hat{p})$")

print("Parametro final:", p_hat.item())

"""## Caso Gaussiana"""

# Parámetros reales 
mu = 10
sd = 100
print("Meida real: ", mu)

# Generar la muestra
N_gauss = 1000
D_gauss = np.random.randn(N_gauss) * sd + mu

# Mostrar distribución
plt.figure(dpi=200)
plt.title("Distribución de la muestra")
sns.distplot(D_gauss)

# Pasar los datos a un tensor de pytorch
D_gauss_t = torch.from_numpy(D_gauss).float()

# Parámetro estimado a optimizar
mu_hat = torch.rand((1,1), requires_grad=True)
print("Media inicial:", mu_hat.item())

# Ciclo de optimizacion por descenso de gradiente
iter = 200
alpha = 0.01

# Guardar el costo en cada paso
c = []

for _ in range(iter):
    # Calcular la función de costo
    l = torch.mean((D_gauss_t - mu_hat).pow(2))

    c.append(l.item())

    # Estimar gradiente de las variables dependientes
    l.backward()

    # Actualizar el valor (se apaga el gradiente)
    with torch.no_grad():
        mu_hat -= alpha * mu_hat.grad

        mu_hat.grad.zero_()

print("Media estimada: ", mu_hat.item())

plt.figure(dpi=200)
plt.plot(c)
plt.title("Error cuadrático medio por época")
plt.xlabel("Iteración")
plt.ylabel("$E[(X-\mu)^2]$")

l.grad

