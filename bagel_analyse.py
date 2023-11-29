#%% Importation des modules --------------------------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
try:
    from bagel_fct import *
except Exception as e:
    print(e)
    pass


#%% Init class ---------------------------------------------------------------------------------------------------------

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
    z = [H, 0]  # [m] Bornes de la position en z
    t = [1 * 60, 3 * 60]  # [sec] Bornes temporels


#%% Paramètres ---------------------------------------------------------------------------------------------------------

prm = Parametre()


#%% graph interpolation température 1 min ------------------------------------------------------------------------------

z, t = position(prm)
T_1min = interpolation_quad(z[:, 0], prm.T[0], prm)
plt.plot(prm.hz, prm.T[0], ".r")
plt.plot(z[:, 0], T_1min)
plt.title("Interpolation de la température à 1 minute")
plt.xlabel("Position en z [m]")
plt.ylabel("Température [K]")
plt.legend(["Valeurs expérimentales", "Interpolation"])
plt.show()


#%% graph interpolation température 3 min (inutile) --------------------------------------------------------------------

# T_3min = interpolation_quad(z[:, 0], prm.T[1], prm)
# plt.plot(prm.hz, prm.T[1], ".r")
# plt.title("Interpolation de la température à 3 minutes")
# plt.xlabel("hauteur z (m)")
# plt.ylabel("Température (K)")
# plt.plot(z[:, 0], T_3min)
# plt.show()


#%% Graphique de la température simulé à 3 min -------------------------------------------------------------------------

T = mdf_assemblage(prm)
plt.plot(prm.hz, prm.T[1], ".r")
plt.title(f"Température simulé de la pâte à 3 minutes (Cp = {prm.Cp}, k = {prm.k}, dt = {prm.dt}, nz = {prm.Nz})")
plt.xlabel("Position en z [m]")
plt.ylabel("Température [K]")
plt.plot(z[:, 0], T[:, -1])
plt.legend(["Valeur expérimentale", "Simulation"])
plt.show()


#%% Graphique de la température simulé color map -----------------------------------------------------------------------

fig,ax = plt.subplots(nrows=1, ncols=1)
ax.set_title(f"Profil de Température simulé en Kelvin (Cp = {prm.Cp}, k = {prm.k}, dt = {prm.dt}, nz = {prm.Nz})")
ax.set_xlabel("Temps [sec]")
ax.set_ylabel("Position en z [m]")
fig1 = ax.pcolormesh(t, z, T)
plt.colorbar(fig1, ax=ax)
plt.show()


#%% Stabilité (Trouver nz et dt)----------------------------------------------------------------------------------------

stab_crit = 0.001
detail = 100
prm.Nz = 100
prm.dt = 0.01
erreur = np.array([])
nz_arr = np.array([])
dt_arr = np.array([])

#%%

T_ref = mdf_assemblage(prm)

#%%

for Nz in np.linspace(3, 100, detail)[:-1]:
    for dt in np.linspace(0.01, (prm.t[1]-prm.t[0])//2 , detail)[1:]:
        prm.Nz = int(Nz)
        prm.dt = dt
        T = mdf_assemblage(prm)
        erreur = np.append(erreur, (np.abs((T[:, -1][0] - T_ref[:, -1][0]) / (T_ref[:, -1][0] - 273.15))))
        nz_arr = np.append(nz_arr, int(Nz))
        dt_arr = np.append(dt_arr, dt)

#%%

ax = plt.figure().add_subplot(111, projection='3d')
ax.bar3d(nz_arr, dt_arr, np.zeros_like(erreur), 1, 1, erreur*100)
ax.set_yscale("symlog")
ax.set_xscale("symlog")
ax.set_title(f"Stabilité de la simulation (Cp = {prm.Cp}, k = {prm.k})")
ax.set_xlabel("Nombre de points en z (nz)")
ax.set_ylabel("Pas de temps (dt) [sec]")
ax.set_zlabel("Stabilité [%]")
ax.set_ylim(0.01)
ax.set_xlim(0.01)
plt.show()

#erreur = erreur.reshape(detail-1,detail-1).transpose()
#nz_arr = nz_arr.reshape(detail-1, detail-1).transpose()
#dt_arr = dt_arr.reshape(detail-1, detail-1).transpose()
#ax = plt.figure().add_subplot(111, projection='3d')
#ax.plot_surface(nz_arr, dt_arr, erreur*100, alpha=0.7)
#ax.set_yscale("symlog")
#ax.set_xscale("symlog")
#ax.set_title(f"Stabilité de la simulation (Cp = {prm.Cp}, k = {prm.k})")
#ax.set_xlabel("Nombre de points en z (nz)")
#ax.set_ylabel("Pas de temps (dt) [sec]")
#ax.set_zlabel("Stabilité [%]")
#ax.set_ylim(0.01)
#ax.set_xlim(0.01)
#plt.show()

erreur_crit = erreur[np.less(erreur, stab_crit)]
prm.Nz = int(nz_arr[np.where(erreur == erreur_crit[np.abs(erreur_crit - erreur_crit.mean()).argmin()])][0])
prm.dt = int(dt_arr[np.where(erreur == erreur_crit[np.abs(erreur_crit - erreur_crit.mean()).argmin()])][0])

print(f"Pour une stabilité à {round(erreur_crit[np.abs(erreur_crit - erreur_crit.mean()).argmin()]*100, 5)} %, on peu utiliser un nz de {prm.Nz} et un pas dt de {prm.dt}")


#%% trouver Cp et k ----------------------------------------------------------------------------------------------------

Cp = np.linspace(700, 1400, 25)
k = np.linspace(0.8, 2.1, 25)
res = np.array([])
for i in Cp:
    for j in k:
        prm.Cp = i
        prm.k = j
        res = np.append(res, np.array([i, j, f_obj(mdf_assemblage(prm), prm)]))

#%%

res = res.reshape(len(Cp)*len(k), 3)
objectif = res[:, 2].reshape(len(Cp),len(k)).transpose()
fig,ax = plt.subplots(nrows=1, ncols=1)
ax.set_title(f"Profil de la fonction-objectif (Nz = {prm.Nz}, dt = {prm.dt})")
ax.set_xlabel("Cp [J/(kg K)]")
ax.set_ylabel("k [W/(m K)]")
fig1 = ax.pcolormesh(Cp, k, objectif)
plt.colorbar(fig1, ax=ax)
plt.show()

prm.Cp = round(float(res[:, 0][np.where(res[:, 2] == np.min(res[:, 2]))]), 4)
prm.k = round(float(res[:, 1][np.where(res[:, 2] == np.min(res[:, 2]))]), 4)

print(f"Pour l'objectif atteint avec {round(float(res[:, 2][np.where(res[:, 2] == np.min(res[:, 2]))])*100, 5)} % d'erreur, on peu utiliser un Cp de {prm.Cp} et un k de {prm.k}")


#%% Graphique de la température simulé à 3 min avec nouveau nz, dt, Cp et k --------------------------------------------

z, t = position(prm)
T = mdf_assemblage(prm)
plt.plot(prm.hz, prm.T[1], ".r")
plt.title(f"Température de la pâte à 3 minutes (Cp = {prm.Cp}, k = {prm.k}, dt = {prm.dt}, nz = {prm.Nz})")
plt.xlabel("hauteur z [m]")
plt.ylabel("Température [K]")
plt.plot(z[:, 0], T[:, -1])
plt.legend(["Valeur expérimentale", "Simulation"])
plt.show()


#%% Graphique de la température simulé color map avec nouveau nz, dt, Cp et k ------------------------------------------

fig,ax = plt.subplots(nrows=1, ncols=1)
ax.set_title(f"Profil de Température en Kelvin (Cp = {prm.Cp}, k = {prm.k}, dt = {prm.dt}, nz = {prm.Nz})")
ax.set_xlabel("Temps [sec]")
ax.set_ylabel("Position en z [m]")
fig1 = ax.pcolormesh(t, z, T)
plt.colorbar(fig1, ax=ax)
plt.show()

# %% Comparaison des interpolation -------------------------------------------------------------------------------------

prm = Parametre()

z, t = position(prm)
T_1min = interpolation_quad(z[:, 0], prm.T[0], prm)
plt.plot(prm.hz, prm.T[0], ".r")
plt.plot(z[:, 0], T_1min)

f = interp1d(prm.hz, prm.T[0], kind='quadratic')
print(prm.z)
xnew = np.arange(prm.z[1], prm.z[0], 0.001)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(xnew, ynew, '--')
plt.title("Interpolation de la température à 1 minute")
plt.xlabel("Position en z [m]")
plt.ylabel("Température [K]")
plt.legend(["Valeurs expérimentales", "Interpolation quadratique", "interpolation de scipy"])
plt.show()
# %%
