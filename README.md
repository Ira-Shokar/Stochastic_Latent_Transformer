## Capturing a Stochastically-Forced Zonal Jet System with a Stochastic Latent Transformer

Seasonal and longer-term prediction of mid-latitude weather and climate remains a major challenge due to the complexity of the mechanisms that drive the dynamics as well as their chaotic nature. This work extend work (<a href="https://www.damtp.cam.ac.uk/user/is500/phd_proposal.html">webpage</a>) which looked at encapsulating the dynamics of a simplified model of atmospheric circulation by using a neural network to find a reduced-order-model of the system. The MRes study highlighted the following research questions, that will form the basis for exploration during the PhD: (i) Can machine learning (ML) be used to better understand the variability exhibited in models of atmospheric circulation? (ii) Can ML be used to emulate improved stochastic parameterisation schemes?

Naturally the two research questions are coupled, as the ability to accurately encapsulate the dynamics of a system will allow us to explore questions regarding its nature, variability, and the tendencies of the parameterisation due to how the model represents these processes, while understanding the variability will inform how best to produce a representation.
          
The project is motivated by the issue of Global Circulation Models (GCMs), that describe oceanic and atmospheric dynamics, being incredibly computationally expensive, and as a result not all scales can be simulated. Processes that take place on length scales smaller than the spatial resolution of the GCM, as well as fast scale dynamics, must be approximated, with these approximations known as sub-grid parameterisations. A large source of model uncertainty is a result of their parameterisation schemes, due to the chaotic nature of the dynamics, leading to questions regarding the fidelity of these models and thus their usefulness.
          
Due to the computational cost of GCMs, the project will focus on a simplified ‘toy’ model, that uses a beta-plane approximation to encapsulate rotational planetary motion, as well reducing the system to a 2-D plane by considering the atmosphere as shallow fluid layer when compared to the horizontal scale. The 2-D nature of the system results in a lack of turbulence and thus a stochastic forcing is required to parameterise the small-scale eddies. This quasi-2-D nature of planetary scale motions gives rise to dynamics of interest, including the formation of jet streams and cyclones.
          
We will focus on the evolution of these jets, that themselves play a fundamental role in the climate system, transporting geophysically-important quantities, such as momentum, heat, and tracers, including chemically and thermodynamically important quantities such as ozone and water vapour. Understanding how these vary in response to changes in concentrations of greenhouse gases is important in understanding the implications of climate change for regional weather patterns. There are outstanding questions regarding factors controlling the variability of these jets, including whether lower or higher resolution models effect the variability of a model as processes that occur at different length scales, such as energy transfers, are required to parameterised.
          
To explore the system variability due to the stochastic parameterisations, addressing (i), ML will be used to find a mapping between a system state and how stable the system is with regards to the attractor that is driving the system. By quantifying stability this way one can begin to understand the impact of the stochastic forcing on the dynamics of the system, and how observed phenomena are driven. This will also be compared to a stratified system to observe how the parameterisations drive the system compare with a system that exhibits its own internally generated turbulence.  We finally will look to develop emulations of stochastic parameterisations using Deep Learning (ii), with the goal of being able to utilise the computational speed-up of ML networks to include more complexity without increasing the time to run whole models- leading to better projections.
