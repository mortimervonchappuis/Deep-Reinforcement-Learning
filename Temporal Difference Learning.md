# Temporal Difference Learning

TD reduces variance, because it excludes future rewards which are more random then the imediate reward.

$$
Q_\pi(s_t, a_t) = \mathbb{E}_\pi[r_t + \gamma Q_\pi(s_{t+1}, a_{t+1})]
$$

The exponentially weighted average leads to:

$$
Q_\pi(s_t, a_t) \leftarrow (1 - \alpha) Q_\pi(s_t, a_t) + \alpha (r_t + \gamma Q_\pi(s_{t+1}, a_{t+1}))
$$

$$
Q_\pi(s_t, a_t) \leftarrow Q_\pi(s_t, a_t) - \alpha Q_\pi(s_t, a_t)+ \alpha (r_t + \gamma Q_\pi(s_{t+1}, a_{t+1}))
$$

$$
Q_\pi(s_t, a_t) \leftarrow Q_\pi(s_t, a_t) + \alpha (r_t + \gamma Q_\pi(s_{t+1}, a_{t+1})  - Q_\pi(s_t, a_t))
$$

$$
Q_\pi(s_t, a_t) \leftarrow Q_\pi(s_t, a_t) + \alpha \delta_t
$$

The standart TD-Error is:

$$
\delta_t \overset{\triangle}{=} r_t + \gamma Q_\pi(s_{t+1}, a_{t+1}) - Q_\pi(s_t, a_t)
$$

TD-Prediction via learning of $V$, TD-Control via learning of $Q$ and $\epsilon$-greedy or Thompson sampling.

$n$-step Estimator:

$$
Q_\pi(s_t, a_t) = \mathbb{E}_\pi\left[\sum_{k = 0}^{n-1} \gamma^k r_{t + k} + \gamma ^{n}Q_\pi(s_{t + n}, a_{t+n}) \right]
$$

$n$-step Return is:

$$
G_{t:t+n} \overset{\triangle}{=} \sum_{k = 0}^{n-1} \gamma^k r_{t + k} + \gamma^n V_\pi(s_{t + n})
$$

$n$-step TD-Error is:

$$
\delta^n_t \overset{\triangle}{=} \sum_{k = 0}^{n-1} \gamma^k r_{t + k} + \gamma^nQ_\pi(s_{t + n}, a_{t+n}) - Q_\pi(s_t, a_t)
$$

$\text{TD}(\lambda)$ Estimator is:

$$
Q_\pi(s_t, a_t) = (1 - \lambda)\mathbb{E}_\pi\left[\sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t + n}\right]
$$

$$
Q_\pi(s_t, a_t) = (1 - \lambda) \left( \mathbb{E}_\pi\left[\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t + n}\right] + \lambda^{T-t-1}G_t \right)
$$

Expected Sarsa is:

$$
Q_\pi(s, a) = \mathbb{E}\left[ r + \gamma \sum_{a'}\pi(a'|s') Q_\pi(s', a') \right]
$$

$n$-step Expected Sarsa is:

$$
Q_\pi (s, a) = \mathbb{E}\left[ \sum_{k=0}^{n-1} \gamma^kr_{t+k} + \sum_{a_{t + n}}\pi(a_{t+n}|s_{t+n})Q_\pi(s_{t+n}, a_{t+n}) \right]
$$

Treebackup is:

$$
G_{t:t+n} \overset{\triangle}{=} R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q(S_{t+1}, a) + \gamma \pi(A_{t+1}|S_{t+1}) G_{t+1:t+n}
$$

$Q(\sigma)$ is:

$$
G_{t:t+n} \overset{\triangle}{=} R_{t+1} + 
\gamma \left \{\begin{matrix} 
G_{t+1:t+n} \quad \text{if} \quad  x < \sigma \quad \text{else}\\
\sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q(S_{t+1}, a) + \pi(A_{t+1}|S_{t+1}) G_{t+1:t+n}
\end{matrix} \right. 
$$

with $x$ as a randomvariable with $x ~ [0, 1]$
