import numpy as np


class StringBuilder:
    """
    Builder class for a Ginzburg-Landau string.
    """

    def __init__(self, F, Nx, Ny, x_lim, y_lim, multicomponent, B_z=0):
        """
        Initialize the StringBuilder.

        Args:
            F   (float): Number of frames.
            Nx  (float): Number of points in the x direction.
            Ny  (float): Number of points in the y direction.
            x_lim (tuple): x-limits of the computational domain.
            y_lim (tuple): y-limits of the computational domain.
            multicomponent (bool): Boolean for multicomponent.
            B_z (float): external magnetic field
        """

        self.F = F
        self.Nx = Nx
        self.Ny = Ny
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.multicomponent = multicomponent
        self.B_z = B_z

        # Coordinate axes
        self.x_ax = np.linspace(self.x_lim[0], self.x_lim[1], self.Nx)
        self.y_ax = np.linspace(self.y_lim[0], self.y_lim[1], self.Ny)
        self.x, self.y = np.meshgrid(self.x_ax, self.y_ax, indexing="ij")

        # Domain definitions
        self.comp_domain_shape_fun = np.vectorize(
            lambda x, y: (
                (x >= self.x_lim[0])
                and (x <= self.x_lim[1])
                and (y >= self.y_lim[0])
                and (y <= self.y_lim[1])
            )
        )
        self.sc_domain_shape_fun = self.comp_domain_shape_fun

        # Global phase difference
        self.phase_fun = lambda x, y: 0

        # Coordinates and windings of vortices in the two components
        self.vortices_1 = []
        self.vortices_2 = []

    def set_comp_domain(self, shape_fun):
        """
        Set the computational domain definition.
        """

        self.comp_domain_shape_fun = shape_fun

    def set_sc_domain(self, shape_fun):
        """
        Set the superconductor domain definition.
        """

        self.sc_domain_shape_fun = shape_fun

    def add_phase_diff(self, phase_fun):
        """
        Add a phase difference between theta_1 and theta_2.
        """
        self.phase_fun = phase_fun

    def add_vortex(self, xv, yv, Nv, component):
        """
        Add a vortex.

        Args:
            xv (tuple)      : X coordinate of the center of the vortex (array of length F).
            yv (tuple)      : Y coordinate of the center of the vortex (array of length F).
            Nv (tuple)      : Winding number of the vortex.
            component (int) : Component in which the vortex is placed.         
        """

        if component == 1:
            self.vortices_1.append([xv, yv, Nv])
        if component == 2:
            self.vortices_2.append([xv, yv, Nv])

    def get_domain_matrices(self):
        """
        Return the matrices of the computational and superconductive domains.
        
        Returns:
            comp_domain (ndarray) : computational domain (Nx, NY) 
            sc_domain (ndarray)   : superconductor domain (Nx, NY)
        """

        comp_domain = np.zeros((self.Nx, self.Ny), dtype=np.intc)
        comp_domain[self.comp_domain_shape_fun(self.x, self.y)] = 1

        sc_domain = np.zeros((self.Nx, self.Ny), dtype=np.intc)
        sc_domain[self.sc_domain_shape_fun(self.x, self.y)] = 1

        return comp_domain, sc_domain

    def generate_matter_fields(self, psi_abs_1=1.0, psi_abs_2=1.0, noise=0.0):
        """
        Generates the initial guess.
        
        Args:
            psi_abs_1 (float, optional): Equilibrium bulk value of psi_1. Defaults to one.
            psi_abs_2 (float, optional): Equilibrium bulk value of psi_2. Defaults to one.
            noise (float, optional): Random noise added to the initial guess. Defaults to zero.

        Returns:
            u_1 (ndarray) : real part of psi_1 field (F, Nx, Ny)     
            v_1 (ndarray) : imag part of psi_1 field (F, Nx, Ny)
            u_2 (ndarray) : real part of psi_2 field (F, Nx, Ny)
            v_2 (ndarray) : imag part of psi_2 field (F, Nx, Ny) 
        """
        sc_domain = np.zeros((self.Nx, self.Ny), dtype=np.intc)
        sc_domain[self.sc_domain_shape_fun(self.x, self.y)] = 1

        #x = np.repeat(self.x[np.newaxis, :, :], self.F, axis=0)
        #y = np.repeat(self.y[np.newaxis, :, :], self.F, axis=0)

        x = self.x
        y = self.y

        u_1 = np.ones((self.F, self.Nx, self.Ny))
        v_1 = np.ones((self.F, self.Nx, self.Ny))

        u_2 = np.ones((self.F, self.Nx, self.Ny))
        v_2 = np.ones((self.F, self.Nx, self.Ny))

        for xv, yv, wv in self.vortices_1:
        
            for N in range(self.F):

                psi_abs = psi_abs_1 + 0 * self.x
                psi_theta = 0 * self.x

                if wv[N] != 0:
                    r_i = np.sqrt(
                        (x - xv[N]) ** 2 + (y - yv[N]) ** 2
                    )
                    psi_abs *= np.tanh(-(r_i ** 2))
                    psi_theta += wv[N] * np.arctan2(
                        y - yv[N], x - xv[N],
                    )

                u_1[N] = (
                    psi_abs * np.cos(psi_theta) + noise * np.random.randn(self.Nx, self.Ny)
                ) * sc_domain
                v_1[N] = (
                    psi_abs * np.sin(psi_theta) + noise * np.random.randn(self.Nx, self.Ny)
                ) * sc_domain

            # Second component
        if self.multicomponent:

            for xv, yv, wv in self.vortices_2:

                for N in range(self.F):

                    psi_abs = psi_abs_2 + 0 * self.x
                    psi_theta = self.phase_fun(self.x, self.y)

                    if wv[N] != 0:
                        r_i = np.sqrt(
                            (self.x - xv[N]) ** 2
                            + (self.y - yv[N]) ** 2
                        )
                        psi_abs *= np.tanh(-(r_i ** 2))
                        psi_theta += wv[N] * np.arctan2(
                            self.y - yv[N], self.x - xv[N]
                        )
                    
                    u_2[N] = (
                        psi_abs * np.cos(psi_theta)
                        + noise * np.random.randn(self.Nx, self.Ny)
                    ) * sc_domain
                    v_2[N] = (
                        psi_abs * np.sin(psi_theta)
                        + noise * np.random.randn(self.Nx, self.Ny)
                    ) * sc_domain

        else:
            u_2 = None
            v_2 = None

        return u_1, v_1, u_2, v_2

    def generate_vector_potential(self, gauge="symmetric"):
        """
        Generates the vector potential field.
        
        Returns:
            a_x (ndarray) : x component of vetor potential (Nx, NY) 
            a_y (ndarray) : y component of vetor potential  (Nx, NY)  
            gauge (str)   : gauge choice, can be "Landau_x", "Landau_y", "symmetric" 
        """

        comp_domain = np.zeros((self.Nx, self.Ny), dtype=np.intc)
        comp_domain[self.comp_domain_shape_fun(self.x, self.y)] = 1

        a_x = np.zeros((self.F, self.Nx, self.Ny))
        a_y = np.zeros((self.F, self.Nx, self.Ny))

        for N in range(self.F):
            if gauge == "Landau_x":
                a_x[N] = -self.B_z * self.y

            elif gauge == "Landau_y":
                a_y[N] = +self.B_z * self.x

            elif gauge == "symmetric":
                a_x[N] = -1 / 2 * self.B_z * self.y
                a_y[N] = +1 / 2 * self.B_z * self.x

            else:
                raise Exception("Not implemented.")

            a_x[N] *= comp_domain
            a_y[N] *= comp_domain

        return a_x, a_y
