# Importation des modules
import numpy as np

def position(z_, t_, prm):
    """ Fonction générant deux matrices de discrétisation de l'espace

    Entrées:
        - Z : Bornes du domaine en z, X = [z_min, z_max]
        - T : Bornes du domaine en t, Y = [t_min, t_max]
        - prm : 
            H [m] Hauteur de la pâte
            rho [kg/m³] Masse volumique de la pâte
            h [W/(m² K)] Coefficient de transfert thermique
            k [W/(m K)] Conductivité thermique
            Cp [J/(kg K)] Chaleur spécifique à pression constante
            Nz Nombre de points de discrétisation
            dt [sec] Pas de temps
            Tp [K] Température du plan chaud
            Ta [K] Température de l'air
            T [K] températures de la pâte au temps t=1min et t=3min
            hz [m] points de mesures

    Sorties (dans l'ordre énuméré ci-bas):
        - z : Matrice (array) de dimension (Nz x Nt) qui contient la position en z
        - t : Matrice (array) de dimension (Nz x Nt) qui contient le temps en t
    """
    Nt = int((t_[1]-t_[0])/(prm.dt))
    # Fonction à écrire
    t = np.zeros((prm.Nz, Nt))
    for i in range(prm.Nz):
        t[i] = np.linspace(t_[0], t_[1], Nt)
        
    z = np.zeros((Nt, prm.Nz))
    for i in range(Nt):
        z[i] = np.linspace(z_[0], z_[1], prm.Nz)
    z = z.T
    return z, t


def interpolation_quad(z, T, prm):
    """ Fonction qui calcul la valeur de température en un
    point selon une interpolation quadratique

    Entrées:
        - z : [m] point de calcul de la température
        - T : [K] températures des points aux alentours
        - prm : 
            H [m] Hauteur de la pâte
            rho [kg/m³] Masse volumique de la pâte
            h [W/(m² K)] Coefficient de transfert thermique
            k [W/(m K)] Conductivité thermique
            Cp [J/(kg K)] Chaleur spécifique à pression constante
            Nz Nombre de points de discrétisation
            dt [sec] Pas de temps
            Tp [K] Température du plan chaud
            Ta [K] Température de l'air
            T [K] températures de la pâte au temps t=1min et t=3min
            hz [m] points de mesures

    Sorties (dans l'ordre énuméré ci-bas):
        - A : Matrice (array)
        - b : Vecteur (array)
    """
    
    T_0 = (z-prm.hz[1])*(z-prm.hz[2])/(prm.hz[0]-prm.hz[1])/(prm.hz[0]-prm.hz[2])
    T_1 = (z-prm.hz[0])*(z-prm.hz[2])/(prm.hz[1]-prm.hz[0])/(prm.hz[1]-prm.hz[2])
    T_2 = (z-prm.hz[0])*(z-prm.hz[1])/(prm.hz[2]-prm.hz[0])/(prm.hz[2]-prm.hz[1])
    
    return T[0]*T_0 + T[1]*T_1 + T[2]*T_2


def mdf_assemblage(z_, t_, prm):
    """ Fonction assemblant la matrice A et le vecteur b

    Entrées:
        - z_ : Bornes du domaine en z, X = [z_min, z_max]
        - t_ : Bornes du domaine en t, Y = [t_min, t_max]
        - prm : 
            H [m] Hauteur de la pâte
            rho [kg/m³] Masse volumique de la pâte
            h [W/(m² K)] Coefficient de transfert thermique
            k [W/(m K)] Conductivité thermique
            Cp [J/(kg K)] Chaleur spécifique à pression constante
            Nz Nombre de points de discrétisation
            dt [sec] Pas de temps
            Tp [K] Température du plan chaud
            Ta [K] Température de l'air
            T [K] températures de la pâte au temps t=1min et t=3min
            hz [m] points de mesures

    Sorties (dans l'ordre énuméré ci-bas):
        - A : Matrice (array)
        - b : Vecteur (array)
    """
    
    z, t = position(z_, t_, prm)
    Nz = prm.Nz
    dz = (z_[1]-z_[0])/prm.Nz
    Nt = int((t_[1]-t_[0])/(prm.dt))
    dt = prm.dt
    k = prm.k
    rho = prm.rho
    Cp = prm.Cp
    Ta = prm.Ta
    h = prm.h
    N = Nz * Nt
    
    T = np.zeros([Nz, Nt])
    
    T[:, 0] = interpolation_quad(z[:, 0], prm.T[0], prm)
    
    for j in range(1, Nt):
        A = np.zeros([Nz, Nz])
        b = np.zeros(Nz)
        for i in range(1, Nz-1):
            A[i][i-1] = -(dt * k)/(dz**2 * rho * Cp)
            A[i][i] = 1 + (2 * dt * k)/(dz**2 * rho * Cp)
            A[i][i+1] = -(dt * k)/(dz**2 * rho * Cp)
            
            b[i] = T[:, j-1][i]
        
        A[0][0] = 2*h*dz/k - 3
        A[0][1] = 4
        A[0][2] = -1
        b[0] = 2*h*dz/k * Ta
        
        A[-1][-1] = 1
        b[-1] = np.linspace(prm.T[0][0], prm.T[1][0], Nt)[j]
        
        T[:, j] = np.linalg.solve(A, b)
        
    return T


def stabilite(T, prm):
    return sum(abs((prm.T[1]-np.array((T[0, -1], T[len(T)//2, -1], T[-1, -1])))/prm.T[1]))