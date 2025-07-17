import numpy as np
import scipy as sp
import itertools
import pyscf

from scipy.special import factorial2

def generate_cartesian_exponents(l):
    """
    Returns a list of (nx, ny, nz) exponent tuples for angular momentum l.
    """
    return [
        (item.count('x'), item.count('y'), item.count('z'))
        for item in itertools.combinations_with_replacement(['x', 'y', 'z'], l)
    ]

def generate_orbitals(mol):
    atomlist = mol._atom
    basislist = mol._basis
    orbital_description = {}
    i = 0

    for atom_index, atom in enumerate(atomlist):  
        atom_name = atom[0]
        atom_coords = atom[1]
        basis = basislist[atom_name]

        for shell in basis:
            l = shell[0]
            coeffs = shell[1:]
            exponents = generate_cartesian_exponents(l)

            for nx, ny, nz in exponents:
                orbital_description[i] = {
                    "Atom": atom_name,
                    "Atom Index": atom_index, 
                    "Coordinates": atom_coords,
                    "Angular Momentum": l,
                    "Exponents": (nx, ny, nz),
                    "Coefficients": coeffs
                }
                i += 1

    return orbital_description, i  # i = total orbitals

def safe_factorial2(n):
    if n <= 0:
        return 1  # define (-1)!! = 1, (0)!! = 1
    return factorial2(n)

def normalize(alpha, l, m, n):
    norm = ((2 * alpha / np.pi)**(3/2) *
            (4 * alpha)**(l + m + n) /
            (safe_factorial2(2 * l - 1) *
             safe_factorial2(2 * m - 1) *
             safe_factorial2(2 * n - 1)))**0.5
    return norm

# See Helgaker 9.2.4

# Normalization factor for a single Gaussian; but we need to multiply for each x, y, z.
# <Gi | Gi > = (2i-1)!!/(4a)^i (pi/2a)**0.5
# <Gi 3d | Gi 3d > = (2i - 1)!! (2j -1)!! (2k - 1)!! / (4a)^(i + j + k) (pi/2a)**(3/2)
# N = 1/sqrt(<Gi | Gi >)

def generateEs(X_PA, X_PB, K00, p, max_l=2):

    # We use the recurrence relations of the Hermite Gaussians here.

    Etij = {}
    X_AB = X_PB - X_PA
    if 0 not in Etij:
        Etij[0] = {}
        Etij[0][(0,0)] = K00

    for i in range(max_l + 1):
        for j in range(max_l + 1):
            for t in range(0, i+j+1):
                if t not in Etij:
                    Etij[t] = {}
                if (i,j) not in Etij[t]:
                    if i == 0 and j == 0 and t == 0:
                        continue
                    if i > 0:
                        term1 = (1/(2*p)) * Etij.get(t-1,{}).get((i-1,j),0)
                        term2 = X_PA * Etij.get(t,{}).get((i-1,j),0)
                        term3 = (t+1) * Etij.get(t+1,{}).get((i-1,j),0)
                        Etij[t][(i,j)] = term1 + term2 + term3
                    elif j > 0:
                        term1 = (1/(2*p)) * Etij.get(t-1, {}).get((i, j-1), 0)
                        term2 = X_PB * Etij.get(t, {}).get((i, j-1), 0)
                        term3 = (t + 1) * Etij.get(t+1, {}).get((i, j-1), 0)
                        Etij[t][(i, j)] = term1 + term2 + term3
    
    return Etij