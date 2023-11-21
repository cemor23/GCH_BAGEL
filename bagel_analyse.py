# Importation des modules
import numpy as np
import matplotlib.pyplot as plt
try:
    from bagel_fct import *
except:
    pass


class Parametre:
    H = 42 / 1000  # [m] Hauteur de la pâte
    rho = 1.07E3  # [kg/m³] Masse volumique de la pâte
    h = 10  # [W/(m² K)] Coefficient de transfert thermique
    k = 0.8  # [W/(m K)] Conductivité thermique
    Cp = 800  # [J/(kg K)] Chaleur spécifique à pression constante
    Nz = 23  # Nombre de points de discrétisation
    dt = 10  # [sec] Pas de temps
    Tp = np.array((34.13, 34.21)) + 273.15  # [K] Température du plan chaud
    Ta = 21.50 + 273.15  # [K] Température de l'air
    T = np.array(((27.92, 24.01, 23.12), (31.73, 25.19, 23.69))) + 273.15  # [K] températures de la pâte au temps t=1min et t=3min
    hz = np.array((0, H/2, H))  # [m] points de mesures


# # Paramètres

prm = Parametre()
z_ = [prm.H, 0]
t_ = [1*60, 3*60]
z, t = position(z_, t_, prm)

# # graph interpolation température 1 min

T_1min = interpolation_quad(z[:, 0], prm.T[0], prm)
plt.plot(prm.hz, prm.T[0], ".r")
plt.plot(z[:, 0], T_1min)
plt.title("Interpolation de la température à 1 minute")
plt.xlabel("hauteur z (m)")
plt.ylabel("Température (K)")
plt.show()

# # graph interpolation température 3 min

T_3min = interpolation_quad(z[:, 0], prm.T[1], prm)
plt.plot(prm.hz, prm.T[1], ".r")
plt.title("Interpolation de la température à 3 minutes")
plt.xlabel("hauteur z (m)")
plt.ylabel("Température (K)")
plt.plot(z[:, 0], T_3min)
plt.show()

T = mdf_assemblage(z_, t_, prm)

# # Graphique color map

fig,ax = plt.subplots(nrows=1, ncols=1)
ax.set_title("Profile de Température (k = 0.8, Cp = 800) en Kelvin")
ax.set_xlabel("Temps (sec)")
ax.set_ylabel("Position en z (m)")
fig1 = ax.pcolormesh(t, z, T)
plt.colorbar(fig1, ax=ax)
plt.show()

# # Stabilité
f_obj = stabilite(T, prm)
print(f"f_obj = {f_obj}")

T = mdf_assemblage(z_, t_, prm)
f_obj = stabilite(T, prm)
f_obj = []
for i in range(3, 50):
    for j in range(3, 50):
        prm.Nz = i
        prm.dt = (t_[1]-t_[0])/j
        f_tmp = stabilite(mdf_assemblage(z_, t_, prm), prm)
        f_obj.append([prm.Nz, prm.dt, f_tmp])

f_obj = np.array(f_obj)
print(f"nz = %s, dt = %s, f_obj = %s" % tuple(f_obj[f_obj[:, 2]==min(f_obj[:, 2])][0]))

Cp = np.linspace(700, 1400, 25)
k = np.linspace(0.8, 2.1, 25)
f_obj = np.array([])
stabToCpk = {}
for i in Cp:
    for j in k:
        prm.Cp = i
        prm.k = j
        f_obj = np.append(f_obj, stabilite(mdf_assemblage(z_, t_, prm), prm))
        stabToCpk.update({stabilite(mdf_assemblage(z_, t_, prm), prm):(i, j)})

f_obj_reshaped = f_obj.reshape(len(Cp),len(k)).transpose()
fig,ax = plt.subplots(nrows=1, ncols=1)
ax.set_title("Profile de stabilité")
ax.set_xlabel("Cp")
ax.set_ylabel("k")
fig1 = ax.pcolormesh(Cp, k, f_obj_reshaped)
plt.colorbar(fig1, ax=ax)
plt.show()
prm.Cp, prm.k = stabToCpk[min(f_obj)]

T = mdf_assemblage(z_, t_, prm)
z, t = position(z_, t_, prm)
fig,ax = plt.subplots(nrows=1, ncols=1)
ax.set_title("Profile de Température en Kelvin")
ax.set_xlabel("Temps (sec)")
ax.set_ylabel("Position en z (m)")
fig1 = ax.pcolormesh(t, z, T)
plt.colorbar(fig1, ax=ax)
plt.show()