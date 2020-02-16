## Portfolio

---

### Research @ NYU Center For Data Science

[Symplectic Integration of Dynamical System with Deep Neural Network](http://example.com/)
<img src="images/non-sep-hamiltonian.png?raw=true"/>
<img src="images/mass_spring_system.png?raw=true"/>


Came up with a framework for learning to integrate Hamiltonian Dynamics through unsupervised learning(where we have access only to the Hamiltonian) and supervised learning(where we have access to data). See above for learned dynamics of 1. a non-separable Hamiltonian and 2. complex mass-spring system with multiple degrees of freedom. Symplecticity is built into the architecture of the neural network using ideas of Normalizing Flow, and the dynamics of a particular Hamiltonian system is learned through minimizing the loss function that encodes the physics of the Hamiltonian system. In the supervised learning setting, the Hamiltonian(or, the physics) of a dynamical system is learned.

---
[Scalable Second Order Method for Non-Convex Optimization](http://example.com/)
<img src="images/opt_loss.png" width = "1000" height = "200" />
<img src="images/opt_grad.png" width = "1000" height = "200" />


A study of optimization algorithms for extremely ill-conditioned, non-convex regression problem where optimization error is crucial. A scalable second order method based on Krylov iteration that attacks the saddle point problem is found to out perform all of the popular first-order methods(e.g. gradient descent, Adam, ...) and other scalable alternatives(e.g. quasi-Newton methods, conjugate gradient method). The plots above compares performance of the Krylov based method with ADAM and the full Newton method on a toy regression problem.

---
