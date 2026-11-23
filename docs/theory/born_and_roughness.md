# Born approximation and roughness PSD

## 2D Kinematic Born Form Factor

For a 2D periodic grating cross-section with complex refractive index contrast $\Delta n(x, z) = -\delta(x, z) + i\beta(x, z)$ over one period $x \in [-\Lambda/2, \Lambda/2)$ and thickness $z \in [0, L_z]$, the 2D kinematic Born scattering amplitude is

$$
F(Q_x, Q_z) = \frac{k_0}{\Lambda\cos\omega} \int_{-\Lambda/2}^{\Lambda/2} dx \int_0^{L_z} dz\,\Delta n(x, z)\,e^{-i(Q_x x + Q_z z)}
$$

Along the $m$-th Bragg truncation rod ($Q_x = m G_0 = 2\pi m / \Lambda$), the Ewald sphere at sample tilt $\omega$ intersects the rod at out-of-plane wavevector

$$
Q_z(m, \omega) = \begin{cases}
\sqrt{k_0^2 - (k_0\sin\omega + m G_0)^2} - k_0\cos\omega & \text{exact Ewald (\texttt{correct})} \\[8pt]
-m G_0 \tan\omega & \text{flat detector (\texttt{small\_angle})}
\end{cases}
$$

and the kinematic diffraction efficiency is $I_m(\omega) = |F(m G_0, Q_z(m, \omega))|^2$.

## 3D Born Form Factor and Self-Affine Line-Edge Roughness

For a 3D grating volume $\Delta n(x, y, z)$ over domain $L_x \times L_y \times L_z$, the 3D Born amplitude is

$$
F(Q_x, Q_y, Q_z) = \frac{k_0}{L_x L_y \cos\omega} \iiint \Delta n(x, y, z)\,e^{-i(Q_x x + Q_y y + Q_z z)}\,dx\,dy\,dz
$$

In [`build_grating_3d`](../api/python/simulation.md#cd_reduce.simulation.structures.build_grating_3d) and [`build_rough_grating_3d`](../api/python/simulation.md#cd_reduce.simulation.structures.build_rough_grating_3d), each line gets a placement $P(y)$ and a width deviation $W(y)$, drawn by [`generate_edge_field`](../api/python/simulation.md#cd_reduce.simulation.structures.generate_edge_field) from two-sided PSD callables `psd_lpr` and `psd_lwr` of $f = Q_y/2\pi$ (in $\mathrm{nm^3}$). The edges are $u_L = P - W/2$ and $u_R = P + W/2$. The Palasantzas shape (`psd_palasantzas`) is

$$
S(f) = 2\sqrt{\pi}\,\xi\,\sigma^2\,\frac{\Gamma(H+\tfrac12)}{\Gamma(H)}\,\left[1 + (2\pi f \xi)^2\right]^{-(H+\frac12)}
$$

with $\sigma^2 = \int S\,\mathrm df$ over both signs of $f$.

The first-order diffuse from rough edges, and how the package inverts it into LPR, LWR and LER PSDs, are described in [Roughness PSDs from diffuse scattering](roughness_psd.md).
