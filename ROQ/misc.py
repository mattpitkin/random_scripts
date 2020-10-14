import numpy as np

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def solarMassToSec(m):

        M = m*4.92686088e-6
        return M

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def m1m2_to_Mc(m1,m2):
    """Chirp mass from m1, m2"""
    return (m1*m2)**(3./5.)/(m1+m2)**(1./5.)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def m1m2_to_nu(m1,m2):
    """Symmetric mass ratio from m1, m2"""
    return m1*m2/(m1+m2)**2

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def m1m2_to_Mcnu(m1, m2):
	"""Compute symmetric mass ratio and chirp mass from m1, m2"""	
	return [m1m2_to_Mc(m1,m2), m1m2_to_nu(m1,m2)]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def q_to_nu(q):
	"""Convert mass ratio (which is >= 1) to symmetric mass ratio"""
	return q / (1.+q)**2.

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def nu_to_q(nu):
	"""Convert symmetric mass ratio to mass ratio (which is >= 1)"""
	return (1.+np.sqrt(1.-4.*nu)-2.*nu)/(2.*nu)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mq_to_m1m2(M, q):
	"""Convert total mass, mass ratio pair to m1, m2"""
	m2 = M/(1.+q)
	m1 = M-m2
	return [m1, m2]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mq_to_Mc(M, q):
	"""Convert mass ratio, total mass pair to chirp mass"""
	return M*q_to_nu(q)**(3./5.)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mcq_to_M(Mc, q):
	"""Convert mass ratio, chirp mass to total mass"""
	return Mc*q_to_nu(q)**(-3./5.)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mcnu_to_M(Mc, nu):
	"""Convert chirp mass and symmetric mass ratio to total mass"""
	return Mc*nu**(-3./5.)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mnu_to_Mc(M, nu):
	"""Convert total mass and symmetric mass ratio to chirp mass"""
	return M*nu**(3./5.)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mcnu_to_m1m2(Mc, nu):
	"""Convert chirp mass, symmetric mass ratio pair to m1, m2"""
	q = nu_to_q(nu)
	M = Mcq_to_M(Mc, q)
	return Mq_to_m1m2(M, q)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def m1m2_to_delta(m1, m2):
	"""Convert m1, m2 pair to relative mass difference [delta = (m1-m2)/(m1+m2)]"""
	return (m1-m2)/(m1+m2)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def delta_to_nu(delta):
	"""Convert relative mass difference (delta) to symmetric mass ratio"""
	return (1.-delta**2)/4.

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def nu_to_delta(nu):
	"""Convert symmetric mass ratio to relative mass difference delta"""
	return np.sqrt(1.-4.*nu)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def X1X2_to_Xs(X1, X2):
	"""Convert dimensionless spins X1, X2 to symmetric spin Xs"""
	return (X1+X2)/2.

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def X1X2_to_Xa(X1, X2):
	"""Convert dimensionless spins X1, X2 to anti-symmetric spin Xa"""
	return (X1-X2)/2.

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def X1X2_to_XsXa(X1, X2):
	"""Convert dimensionless spins X1, X2 to symmetric and anti-symmetric spins Xs, Xa"""
	return [X1X2_to_Xs(X1,X2), X1X2_to_Xa(X1,X2)]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def fgwisco(Mtot):

	Mtot = solarMassToSec(Mtot)
	"""GW frequency at ISCO. [Note: Maggiore's text has an extra 1/2.]"""
	return 6.0**(-1.5) / (np.pi*Mtot)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def chebyshev_gauss_lobatto_nodes_and_weights(int_min, int_max, num_nodes):
                """Chebyshev Gauss-Lobatto quadrature rule with num nodes"""
                n = int(num_nodes)-1.
                nodes = np.array([-np.cos(np.pi*ii/n) for ii in range(int(num_nodes))])
                weights = np.pi/n * np.sqrt(1.-nodes**2.)
                weights[0] /= 2.
                weights[-1] /= 2.
                return [nodes*(int_max-int_min)/2.+(int_max+int_min)/2., weights*(int_max-int_min)/2.]

def dot_product(weights, a, b):

	assert len(a) == len(b)
	return np.vdot(a*weights, b)
 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def project_onto_basis(integration_weights, e, h, projections, proj_coefficients, iter):

	for j in range(len(h)):
		proj_coefficients[iter][j] = dot_product(integration_weights, e[iter], h[j])
		projections[j] += proj_coefficients[iter][j]*e[iter]
	return projections

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def B_matrix(invV, e):

        B_matrix = np.inner(invV.T, e[0:(invV.shape[0])].T)
        return B_matrix

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def emp_interp(B_matrix, func, indices):

        # B : RB matrix
        assert B_matrix.shape[0] == len(indices)

        interpolant = np.inner(func[indices].T, B_matrix.T)

        return interpolant 
