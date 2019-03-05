# Boiler
Scientific project
----------------------
At this very first work we select vapor quality for searching of appropriate probability equation. The reason is following, vapor quality easy to observe with optical methods and bubbles quite descriptive to understand variational approach.  In fact, we are searching for complex function describing probability \{0;1\} of volume fraction for each point of water tank. In particular, there is a conditional probability for each pixel of image in case of known state of all pixels at previous moment of time

![Probability equation](https://github.com/rumbok/Boiler/blob/master/pictures/probability.png?raw=true)

  where x pixel value for every image part, t -time, z- number of random variables with the normal distribution. The value of each pixel $x_{i} $ derived of particular liquid volume fraction attributed as 0 if 100\% vapour or 1 i 100\%liquid observed. The number of random variables present influene of chaotic nature. This is adjusting value for optimization. Moreover, this is a part of parametrization of VRNN method[]. In addition, no one knows how much random source observed in particular boiling regime.
 The probability is approximating for region near the heating wall. We have made this constrain because we gain the attention to heattransfer. In case of calculating flow vapour quality area location and size should be varied.
