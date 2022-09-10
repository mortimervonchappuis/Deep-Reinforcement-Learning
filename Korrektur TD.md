$$
S = (1-\lambda)\sum_{n=1}^{T-t-1} [\lambda^{n-1}] + \lambda^{T-t-1} = (1 - \lambda)S_0 + \gamma^{T-t-1}
$$

$S_0$ auflösen ergibt sich wie folgt:

$$
S_0 = \sum_{n=1}^{T-t-1} \lambda^{n-1} = \sum_{n=0}^{T-t-2} \lambda^{n} 
$$

$$
S_1 = \gamma \cdot S_0 = \sum_{n=1}^{T-t-1} \lambda^{n} 
$$

$$
S_0 - S_1 = S_0 - \gamma S_0 = \sum_{n=0}^{T-t-2} \lambda^{n} - \sum_{n=1}^{T-t-1} \lambda^{n} = 1 - \gamma^{T-t-1} 
$$

$$
S_0 (1 - \gamma) = 1 - \gamma^{T-t-1}
$$

$$
S_0 = \frac{1 - \gamma^{T-t-1}}{1-\gamma}
$$

Durch einsetzten von $S_0$ folgt:

$$
S = \cancel{(1 - \gamma)} \cdot \frac{1 - \gamma^{T-t-1}}{\cancel{1-\gamma}} + \gamma^{T-t-1}
$$

$$
S = 1 - \gamma^{T-t-1} + \gamma^{T-t-1} = 1
$$


