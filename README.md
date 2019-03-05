# Boiler
Scientific project
----------------------
The relevance of the neural network application for the pool boiling analysis is investigating. Machine learning methods are used for searching the hidden dependences between local and average parameters. The vapor quality used as a local parameter as far as the heating electrical current and temperature as the average ones.

It is known, that the boiling of liquid is one of sophisticated processes in universe. The heat transfer process delivers an energy from a hot solid surface to a cold liquid. Boiling is attractable approach of heat transfer due to hight heat transfer coefficient. This is signify that boiling based devices are lighter and smaller then the others more than ten times.

It haves a good chances that this process has chaotic nature. That is why variational approach can be more appropriate than direct numerical simulations or empirical equations.

At this very first work we have selected vapor quality for searching of appropriate probability equation. The reason is following, the vapor quality is easy to observe with optical methods and bubbles quite descriptive for understanding a variational approach.  In fact, we are searching for complex function describing probability {0;1} of volume fraction for each point of water tank. In particular, there is a conditional probability for each pixel of image in case of known state of all pixels at previous moments of time

![Probability equation](https://github.com/rumbok/Boiler/blob/master/pictures/probability.png?raw=true)

  where x pixel value for every image part, t -time, z- number of random variables with the normal distribution. The bright value of each pixel x is derived of particular liquid volume fraction attributed as 0 in case of 100% vapor or 1 in case of 100% liquid. 
  
For better understanding of variational approach look at some extreme cases. Firstly, we have some heat load (for example 40 kW/m2, 1 bar, water). This is regime with the mostly pulsating nucleation sites. Sometimes the observing region does not have vapor bubbles at all.
Thus, the probability of rising the volume fraction for each pixel in front of the heater must be maximal (fig.1a). On the contrary, such probability for the remote pixels will be minimal (fig.1b). 

![Probability distribution no bubbles](https://github.com/rumbok/Boiler/blob/master/pictures/no_bubble_Pr.png?raw=true)

Fig.1. Imagination of probability function of getting area dark. 40kW/m2

Secondly, consider the boiling under 80 kW/m2 with same other conditions. In that case a lot of bubbles are rising simultaneously. If the number of presented bubbles enough for handle particular heat load, there are no reasons to appears a new bubble at the moment. Thus, the probability of becoming dark for the light pixels will be low for area nearby heater surface (fig.2b). For the other hand, the light pixel will be painted to dark with high probability if it located on the front of rising bubbles (fig.2a). 

![Probability equation](https://github.com/rumbok/Boiler/blob/master/pictures/many_bubble_Pr.png?raw=true)

Fig.2. Imagination of probability function of getting area dark. 80kW/m2

Distribution of vapor quality for an arbitrary pixel has two maximums near zero and one (fig.3). Thus, Log error generates a huge loss value near zero and one. Therefore, MSE is more suitable and the learning process becomes faster.

![Probability equation](https://github.com/rumbok/Boiler/blob/master/pictures/Distribution.png?raw=true)

Fig.3. Distribution of an arbitrary pixel bright on screen. Original vs Model

Currently, we are comparing different computer vision neural networks architectures with purpose of prediction a local parameters behaviour.
