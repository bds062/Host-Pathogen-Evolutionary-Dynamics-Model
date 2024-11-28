This project represents a host pathogen evolutionary model for the ecological dynamics between Silene latifolia and Anther Smut.

This repository contains the code used to generate population graphs and raster graphs for this model, along with the equations and variables themselves being used.

The model itself is represented by the following system of differential equations:

$$\begin{align*}
\dot{S_j} &= b(t)S_f - S_j(\beta_j I_f + \gamma N + m + \mu)\\
\dot{S_v} &= m S_j + v(t) S_f - S_v(\beta_v  I_f + f(t) + \mu)\\
\dot{S_f} &= f(t) S_v - S_f(\beta_f I_f + v(t) + \mu)\\
\dot{I_j} &= \beta_j S_j I_f - I_j(m + \mu)\\
\dot{I_v} &= m I_j + \beta_v S_v I_f  + v(t)I_f - I_v(f(t) + \mu)\\
\dot{I_f} &= f(t) I_v + I_f(\beta_f S_f - v(t) - \mu) 
\end{align*}$$

The system can also be represented by the following diagram:

<img width="851" alt="image" src="https://github.com/user-attachments/assets/faf4d0fd-8da9-450b-a370-a3bf5e64c31f">



For example:

![image](https://github.com/user-attachments/assets/b5abef7a-3445-481b-a6ac-7f4087dcba5f)

<img width="698" alt="image" src="https://github.com/user-attachments/assets/a7a4c725-8758-4a50-83b0-7a8e73cf53ec">

<img width="695" alt="image" src="https://github.com/user-attachments/assets/202ce9b2-5d57-4a27-b301-4b9bb3dc1392">
