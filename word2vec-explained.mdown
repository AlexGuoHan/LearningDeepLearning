
## Introduction
Neural Language Probabilistic Language Model specifies rge distribution for the target words given the a sequence of words $h$ from content  
  
$$ P_{\theta}^{h}(w) = \frac{exp(s_\theta (w,h)} {\sum exp(s_\theta (w, h))}$$

This equation is intractable, and three solutions are presented:
1. tree structured vocabulary --> nontrival
2. importance sampling --> instable
3. NCE


## Scalable Log-Bilinear Models
let $q_w$ and $\gamma_w$ be the target and content representations for word $w$, given a sequence of context words $h = w_1, ..., w_n$, the model computes the predicted representation for the target word by $\widehat{q} = \sum c_i \odot \gamma_w$ where $c_i$ is the weight, and $s_\theta(w,h) = \widehat{q}_h ^T q_w + b_w$  
  
As our main concern is learning word representations, we are free to move away from the paradigm of predicting the target from context and do the reverse. This is motivated by the **distributed hypothesis**, words with similar meaning often occur in similar contexts. Thus we'll be looking for word representations that capture the context distributions -- to predict context from words

Unfortunately, predicting n-word context requires modeling the joing distribution of n-words. This is considerably harder than modeling one of the words. We can make this trackable by assuming words in different context positions are conditionally independent: given current word $w$

$$P_\theta^w(h) = \prod p_{i,\theta}^w(w_i) $$
$$s_{i,\theta}(w_i,w) = (c_i \odot \gamma_w)^T q_{w_i} + b_{w_i}$$

where the $c_i$, position-specific weight is optional


