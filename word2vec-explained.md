
<!-- MarkdownTOC depth=2 autoanchor='true' autolink='true' bracket='round' -->

- [Noise Constrastive Estimation](#noise-constrastive-estimation)
    - [Introduction](#introduction)
    - [Scalable Log-Bilinear Models](#scalable-log-bilinear-models)
    - [Noise Constrastive Estimation](#noise-constrastive-estimation-1)

<!-- /MarkdownTOC -->

<a name="noise-constrastive-estimation"></a>
# Noise Constrastive Estimation
<a name="introduction"></a>
## Introduction
Neural Language Probabilistic Language Model specifies rge distribution for the target words given the a sequence of words $h$ from content  
  
$$ P_{\theta}^{h}(w) = \frac{exp(s_\theta (w,h)} {\sum exp(s_\theta (w, h))}$$

This equation is intractable, and three solutions are presented:
1. tree structured vocabulary $\Rightarrow$ nontrival
2. importance sampling $\Rightarrow$ instable
3. NCE


<a name="scalable-log-bilinear-models"></a>
## Scalable Log-Bilinear Models
let $q_w$ and $\gamma_w$ be the target and content representations for word $w$, given a sequence of context words $h = w_1, ..., w_n$, the model computes the predicted representation for the target word by $\widehat{q} = \sum c_i \odot \gamma_w$ where $c_i$ is the weight, and $s_\theta(w,h) = \widehat{q}_h ^T q_w + b_w$  
  
As our main concern is learning word representations, we are free to move away from the paradigm of predicting the target from context and do the reverse. This is motivated by the **distributed hypothesis**, words with similar meaning often occur in similar contexts. Thus we'll be looking for word representations that capture the context distributions -- to predict context from words

Unfortunately, predicting n-word context requires modeling the joing distribution of n-words. This is considerably harder than modeling one of the words. We can make this trackable by assuming words in different context positions are conditionally independent: given current word $w$

$$P_\theta^w(h) = \prod p_{i,\theta}^w(w_i) $$
$$s_{i,\theta}(w_i,w) = (c_i \odot \gamma_w)^T q_{w_i} + b_{w_i}$$

where the $c_i$, position-specific weight is optional

<a name="noise-constrastive-estimation-1"></a>
## Noise Constrastive Estimation
NCE is a method for fitting unnormalized methods, based on the reduction of density estimation to probability binary classification. The basic idea is to train a logistic regression classifier to discriminate between samples from the data distribution and samples from noise distribution. This is done based on the ration of $P_{x \sim model}$ and $P_{x \sim noise}$. NCE allows us to fit models that are not explicitly normalized, making the training time effectively independent of vocabulary size.  

Suppose we want to learn the distribution of words for some specific context $h$, or $P^h (w)$. To do that, we create an auxiliary binary classification problem, treating data as positive, and samples from a noise distribution $P_n (w)$ as negative. In the original research, authors use *global unigram distribution* of training data as noise distribution. 

If we assume that noise samples are $k$ times more frequent than data samples, the probability that given sample come from data $P^h (D=1 | h) = \frac{P^h_d(w)}{P^h_d(w) + kP_n(w)}$. We can estimate this probability using the model distribution in place of $P^h_d$, that is:

$$ P^h(D=1 | w,\theta) = \frac{P^h_\theta(w)}{P^h_\theta(w) + kP_n(w)} = \sigma(\Delta S_\theta (w,h)$$

where the $P^h_\theta(w)$ is model distribution, $\sigma(\cdot)$ is logistic function, and $\Delta S_\theta (w,h) = S_\theta(w,h) - log(kP_n(w))$ is the difference in the scores of word $w$ under model distribution and scaled noise distribution. Note that the equation used $S_\theta(w,h)$ instead of $log P_\theta^h(w)$, ignoring the normalization term since we are dealing with unnormalized models. Remember that NCE model's objective encourages the model to be appropriately normalized and recovers a perfectly normalized model isf the model class contains the data distribution.

The model is fit by maximizing the expected log posterior probability of correct label D:

$$J^h(\theta) = \mathbb{E}_{x\sim P^h_d}[log P^h(D=1 | w,\theta)] + k \mathbb{E}_n log P^h(D=0 | w,\theta)]$$

$$=\mathbb{E}_{P^h_d}[log \sigma(\Delta S_\theta(w,h))] + k\mathbb{E}_{P_n}[log(1-\sigma(\Delta S_\theta(w,h)))]$$
