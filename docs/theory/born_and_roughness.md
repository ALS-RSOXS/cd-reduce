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

In [`build_grating_3d`](../api/python/simulation.md#cd_reduce.simulation.structures.build_grating_3d), left and right sidewall displacements $\Delta x_{L,R}(y)$ are synthesized from the self-affine Palasantzas power spectral density:

$$
\mathrm{PSD}_{\mathrm{LER}}(Q_y) = \frac{2\sigma_{\mathrm{LER}}^2 \xi_y}{\bigl(1 + a (Q_y \xi_y)^2\bigr)^{1/2 + H}}, \qquad a = \frac{\sqrt{\pi}\,\Gamma(H + 1/2)}{2\,\Gamma(H)}
$$

In the perturbative Born regime ($Q_x \sigma_{\mathrm{LER}} \ll 1$), the off-specular diffuse sheet intensity around the $m$-th Bragg order ($Q_x = m G_0$) normalized to the integrated Bragg order intensity $I_m$ is independent of optical contrast, height, and $Q_z$ form factor:

$$
\frac{1}{I_m}\frac{dI_{\mathrm{diff}}}{dQ_x\,dQ_y}\bigg|_{Q_x = m G_0} = \frac{2\Lambda\,\bigl[1 + c\cos(m G_0 w)\bigr]}{(2\pi)^2\,\bigl|F_m^{(x)}(w)\bigr|^2}\,S_{\mathrm{LER}}(Q_y), \qquad F_m^{(x)}(w) = \frac{2\sin(m G_0 w / 2)}{m G_0}
$$

where $w$ is the line width and $c \in [-1, 1]$ is the cross-correlation coefficient between left and right sidewall displacements ($c = +1$ for pure LER, $c = -1$ for pure symmetric LWR, and $c = 0$ for uncorrelated sidewalls). Setting $c = 0$ yields the parameter-free Born inversion implemented in [`readoff_edge_psd_born`](../api/python/reduction.md#cd_reduce.reduction.mapping.readoff_edge_psd_born).
