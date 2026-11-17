# Split-step Fourier multislice

## Paraxial Wave Equation in the Tilted Sample Frame

In the scalar Helmholtz equation $\bigl[\nabla^2 + k_0^2 n^2(\mathbf{r})\bigr] E(\mathbf{r}) = 0$ with $n(\mathbf{r}) = 1 - \delta(\mathbf{r}) + i\beta(\mathbf{r})$ and $|\delta|, |\beta| \ll 1$, a plane wave incident at sample tilt angle $\omega$ enters the top surface ($z = 0$) and propagates toward $z = L_z$ along wavevector $(k_{x,\mathrm{inc}}, 0, k_{z,\mathrm{inc}}) = (k_0\sin\omega, 0, k_0\cos\omega)$.

Dividing the structure into $N_z$ thin slices of thickness $\Delta z = L_z / N_z$, the Split-Step Fourier (beam-propagation / Cowley–Moodie multislice) method alternates between real-space phase-and-absorption transmission and Fourier-space free-space propagation across each slice $k = 0, \dots, N_z - 1$:

### 1. Half-step free-space propagation ($\Delta z / 2$)

In transverse Fourier space $(q_x, q_y)$, each spatial frequency mode accumulates the tilted angular-spectrum phase over $\Delta z / 2$:

$$
\tilde{\psi}(q_x, q_y, z_k^-) = \mathcal{F}_{x,y}\bigl[\psi(x, y, z_k)\bigr]\,P\!\left(q_x, q_y, \frac{\Delta z}{2}\right)
$$

where the propagator kernel in the sample frame is:

$$
P(q_x, q_y, \delta z) = \exp\!\left[-i\,\delta z\left(q_x \tan\omega + \frac{q_x^2 + q_y^2}{2 k_0 \cos\omega}\right)\right]
$$

Here $-i\,\delta z\,q_x\tan\omega$ shifts the beam laterally by $\delta z \tan\omega$ as it traverses the tilted sample slice, and $-i\,\delta z\,(q_x^2 + q_y^2) / (2 k_0 \cos\omega)$ accounts for Fresnel diffraction (Ewald sphere curvature).

### 2. Thin-slice phase-grating transmission ($\Delta z$)

Returning to real space via inverse FFT, the wavefield is multiplied by the complex transmission function of slice $k$:

$$
\psi(x, y, z_k^+) = \mathcal{F}_{x,y}^{-1}\bigl[\tilde{\psi}(q_x, q_y, z_k^-)\bigr]\,T_k(x, y)
$$

$$
T_k(x, y) = \exp\!\left[-\frac{i k_0 \Delta z}{\cos\omega}\,\delta(x, y, z_k) - \frac{k_0 \Delta z}{\cos\omega}\,\beta(x, y, z_k)\right]
$$

### 3. Second half-step propagation ($\Delta z / 2$)

A second half-step propagation $P(q_x, q_y, \Delta z / 2)$ completes the symmetric Strang splitting for slice $k$, giving $\mathcal{O}(\Delta z^3)$ local accuracy:

$$
\psi(x, y, z_{k+1}) = \mathcal{F}_{x,y}^{-1}\!\left\{\mathcal{F}_{x,y}\bigl[\psi(x, y, z_k^+)\bigr]\,P\!\left(q_x, q_y, \frac{\Delta z}{2}\right)\right\}
$$

## Exit Wave Projection and Dynamical Effects

At the exit plane $z = L_z$, the 1D or 2D Fourier transform of the exit wave $\psi_{\mathrm{exit}}(x, y) - \langle\psi_{\mathrm{exit}}\rangle$ yields the complex diffraction amplitudes $A_m(\omega)$ at the grating harmonics $Q_x = m G_0$:

$$
I_m(\omega) = |A_m(\omega)|^2
$$

Because $T_k(x, y)$ attenuates the wavefield at every depth $z_k$ when $\beta > 0$ and refracts the phase front when $\delta \neq 0$, the multislice solver naturally captures:

- **Kinematic convergence**: As $\delta, \beta \to 0$, $\exp\bigl[-i k_0 \Delta z (\delta - i\beta)/\cos\omega\bigr] - 1 \approx -i k_0 \Delta z (\delta - i\beta)/\cos\omega$, and the sum over slices reduces identically to the 2D/3D Born Riemann integral.
- **Dynamical attenuation**: Strong absorption ($\beta \gtrsim 10^{-3}$) damps the wavefield inside the lines, reducing overall diffraction efficiency relative to the unattenuated Born form factor.
- **Friedel asymmetry ($I_{+m}(\omega) \neq I_{-m}(\omega)$)**: For structures with depth-dependent asymmetry (such as tapered sidewalls, top corner rounding, or bilayer caps) combined with absorption ($\beta > 0$), multiple scattering and depth-dependent attenuation break Friedel's law between $+m$ and $-m$ orders.
