# Sequence to Sequence Model Theory Review
<!-- toc orderedList:0 depthFrom:1 depthTo:1 -->

* [Sequence to Sequence Model Theory Review](#sequence-to-sequence-model-theory-review)
* [Theory Review](#theory-review)
* [Neural Machine Translation By Jointly Learning To Align and Translate](#neural-machine-translation-by-jointly-learning-to-align-and-translate)
* [Densely Connected Convolutional Networks](#densely-connected-convolutional-networks)
* [ADAM: A Method For Stochastic Optimization](#adam-a-method-for-stochastic-optimization)
* [Neural Machine Translation of Rare Words with Subword Units](#neural-machine-translation-of-rare-words-with-subword-units)

<!-- tocstop -->



# Theory Review
[Massive Exploration of Neural Machine Translation](https://arxiv.org/pdf/1703.03906.pdf)

This paper introduces the architectures for the following deep learning models, are acts as the reading roadmaps for myself.

**Some Topics and Keywords are listed below**

Topics | Related Topics
:--- |:---
Moses |
Adam Optimizer |
Byte Pair Encoding |
LSTM | Gated Recurrent Units
Attention | Multiplicative Attention
Residual Connections | Dense Residual Connections
Bi-Directional RNN | Unidirectional RNN
Beam Search | Length Normalization




* [Attention Mechanism](https://arxiv.org/pdf/1409.0473.pdf)
* [Adam Optimizer](https://arxiv.org/pdf/1412.6980.pdf)
* [Byte Pair Encoding](https://arxiv.org/pdf/1508.07909.pdf)
* [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)



### Embedding Dimensionality
2048 dimensional embeddings yielded the overall best results, 128 dimensional embeddings performed sufficiently well and fast

### RNN Cell Variant
A motivation for gated cells like GRU and LSTM is the vanishing gradient. Using vanilla RNN cells, deep networks cannot efficiently propagate information and gradients through multiple layers and time steps. But with an attention-based model, it is believed that decoder should be able to make decisions based on the current input and the attention context. It is hypothesized that gating mechanism is not strictly necessary.
<span style="color:RED">However</span>, LSTM consistently outperformed GRU cells, and vanilla decoder is unable to learn as well as gated variants. This suggests that **decoder indeed pases information in its own state through multiple time steps instead of relying on the attention mechanism**

### ENcoder Decoder Depth
**Encoder**: No clear evidence is found that encoder depth beyond two layers is necessary, but deeper models with residual connections to be significantly more likely to diverge during training. The best deep residual models achieved good results, but only $\frac{1}{4}$ converged.

**Decoder**: Deeper models putperformed shallower ones by a snall margin. It is impossible to to train decoders with 8+ layers without residual connecitons. Across deep decoder experiments, dense residuak connecitons consistently outperformed regular residual connections and converged much faster in terms of step count.

Deep models are expected to perform better acorss the board, but more robust techniques for optimizing deep sequential models are in need.


### Unidirectional and Bi-Directional Encoder
Bi-Directional encoders are able to create representations that take into account both past and future inputs, while Unidirectional encoders can only take past input. But the computation of Unidirectional encoders can be easily paralleized on GPUs.

Experiments suggest that Bi-Directional encoders generally outperformed Unidirectional encoders, but not by a large margin. The encoders with reversed source consistently outperformed their non-reversed counterparts, but not shallwer Bi-Directional encoders.

### Attention Mechanisn
The two most commonly used attention mechanism are the **additive** variant, and the computationally less expensive **Multiplicative** variant.

It is found that parameterized additive attention mechanism slightly but consistently outperformed the multiplicative counterpart, with the attention dimensionality having little effect. Models without attention mechanism bahaves more than poorly, and models with attention exhibited significantly larger gradient update, acting like a "weighted skip connection" that optimizes gradient flow more than like a "memory" that allow encoder to access source states.

### Beam Search Strategies
Beam search is a commonly used technique to find target sequences that maximize some scoring function through three search. Here is score is the log probability of target sequence given source. Recently, extensions such as coverage penalties and length normalization have been shown to imporve decoding results.

A well-tuned beam search is crecial to achieving good results consistently.

Hyperparameter | Value
:---|:---
Embedding dim | 512
RNN cell variant | LSTM
Encoder depth | 4
Decodner depth | 4
Attention dim | 512
Attention type | Additive
Encoder type | bidirectional
Beam size | 10
Length penality | 1.0



# Neural Machine Translation By Jointly Learning To Align and Translate
## Introduction
Most of of the proposed neural machine translation models belong to a family of _encoder decoder_, with an encoder and a decoder for each language, or involve a language-specifid encoder applied to each sentence whose outputs are then compared. AN encoder reads and encodes a source sentence into a fixed-length vector, and a decoder then ouputs a translation.

A potential issue is that compressing all information into a fixed-length vector makes it difficult for the betwork to cope with long sentences. Attention is an extention to the encoder decoder model that learns to align and translate jointly. Each time the model generates a word in a translation, it searches for a set of position in a source sentence where the most relevant information is concentrated. The model then predicts a target word based on the context vectors associated with these source positions and all previous generated target workds. It chooses a subset of these vectors adaptively while decoding the translation.

## Background
Translation is equivalent to finding a target sentence $y$ that maximizes the conditional probability of $y$ given a source sentence $x$, $\arg\max_y p(y|x)$, and the conditional probability is parameterized by a model. The model have two components, encoder and decoder.

An **encoder** reads the input sentence, a sequence of vectors $x=(x_1, ..., x_{T_x})$ into a vector $c$. The most common approach is to use RNN such that

$$h_t = f(x_t, h_{t-1})$$ $$c=q({h_1, ..., h_{T_x}})$$
where $h_t \in \mathbb{R}^n$ is a hiddens state at time $t$ and $c$ is a vector generated from the sequence of hidden states. $f,q$ are nonlinear equations, such as LSTM.

An **decoder** is often trained to predict the next word $y_{t'}$ given the context vector c and previously predicted words ${y_1, ..., y_{t'-1}}$. The decoder defines a probability over the translation $y$ by decomposing the joint probability into ordered conditinals: $$ p(y) = \prod_t^T p(y_t | \{y_1, ..., y_{t-1}\}, c) $$ where $y = (y_1, ..., y_{T_y})$. With an RNN, each conditional probability is modeled as $$p(y_t | \{y_1, ..., y_{t-1}\},c) = g(y_{t-1}, s_t, c)$$ for $g$ a nonlinear, potentially multi-layered function with $s_t$ the hidden state of RNN.

## Learning to Align and Translate
The proposed architecture consists of a bidirectional RNN as an encoder and a decoder that emulates searching through a source sentence during decoding a translation.

### Decoder
The conditional probability is defined: $$p(y_i | y_1, ..., y_{i-1}, x) = g(y_{i-1}, s_i, c_i)$$ $$ s_i = f(s_{i-1}, y_{i-1}, c_i)$$
where $s_i$ is an RNN hidden state, and a distinct context vector $c_i$ for each target word $y_i$. The context vector $c_i$ depends on a sequence of _annotations_ $h_1, ..., h_{T_x}$ to which an encoder maps the input sentence. Each annoation $h_i$ contains information about the whole input sequence with a strong focus on oarts surrounding $i$-th word of the input senence.

Given the annotations, we can compute context vector as weighted sum: $$ c_i = \sum_j^{T_x} \alpha_{ij}h_j$$ with $\alpha_{ij}$ of each annotation computed by $$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum^{T_x}_k \exp(e_{ik})}$$ with $$ e_{ij} = a(s_{i-1}, h_j)$$ an alignment model that scores how well the inputs around position $j$ and the output at position $i$ match. The score is based on the RNN hidden state $s_{i-1}$ and $j$-th annotation $h_j$ of the input sentence.

Putting them together:
$$ \text{context}\space c_i = \sum_j^{T_x} \alpha_{ij}h_j$$ $$\Uparrow$$ $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum^{T_x}_k \exp(e_{ik})} \space\textit{weight for each annotation}$$ $$\Uparrow$$ $$e_{ij} = a(s_{i-1}, h_j)  \space\textit{alignment model}$$

The alignment model $a$ is parameterized as a feedforward neural network. This is trained jointly with the rest parameters, and computes a soft alignment to allow gradient of cost function to be backpropagated through. The approach of taking a weighted sum of all the annotations can be understood as computing an expected annotation, over possible alignment. Let $$ \alpha_{ij} = p_\text{target \textit{yi} is aligned to source \textit{xj}}$$  $$c_i = \sum p_{alignment} \space \cdot h_j=\mathbb{E}[h_j]$$

The importance, relative weight, of annoation $h_j$ is reflected by $\alpha_{ij}$, parameterized by energy $e_{ij}$, and is dependent on previous hidden state $s_{i-1}$. Note that $s_{i-1}$  decides the next state $s_j$ and generates $y_i$. Intuitively, this implements a mechanism of attention in the decoder, and it decides parts of the source sentence to pay attention to.


### Encoder
In the proposed scheme, we would like the annotations of each word to summarize not only the preceeding words, but also the following words. Hence a bidirectional RNN is proposed.

The forward $\vec{f}$  reads input sequence, and calculates a sequence of forward hiddent states $(\vec{h_1^f}, ..., \vec{h_{T_x}^f})$. The backward RNN $\vec{b}$ reads the sequence in reverse order $(\vec{h_1^b}, ..., \vec{h_{T_x}^b})$. The annotation for each word can be concatenated them $h_j = [\vec{ {h^f_j}^T}; \vec{ {h^b_j}^T}]^T$. In this way, the annotation $h_j$ contains the summaries of both the preceeding words and following words. The nature of RNN will better represent recient inputs, the annotaion hence will be focused on the words around $x_j$. This sequence of annoation is used by the decoder and the alignment model to compute the context vector.


# Densely Connected Convolutional Networks
![denseResNet](/assets/denseResNet.png)

> DenseNets alleviate the vanishing gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.

## Introduction
As CNNs become increasingy deep, information (input or gradient) can vanish by the time it reaches the end (beginning) of the network. Recent works create short paths from early layers to later layers to address this issue. In this research, the authors connect all layers (with mathcing feature-map sizes) with each other to ensure maximum information flow between layers in the network. Each layer obtains additional inputs from all preceeding layers and passes on its information to all subsequent layers. This introduce $\frac{L(L+1)}{2}$ connections in $L$-layer network.

As there is no need to re-learn redundant feature maps, DenseNet requires fewer parameters. Recent studies show that many layers in ResNet contribute very little and can be randomly dropped during training. The DenseNet explicitply differentiates between information added to the network and information preserved. Hence its layers are very narrow, adding only a small set of feature-maps to the "collective knowledge"  of the network and keep the remaining feature maps unchanged. It also has improved flow of information and gradients throughout the network, making them easy to train. Each layer has direct acess to the gradients from the loss function and original input signal. It aksi gas a regularizing effect, reducing over-fitting with smaller training sets.

Instead of drawing representational power from extremely deep or wide architectures, DenseNet exploit the potential of network through feature reuse, yielding condensed models easy to tran and highly parameter-efficient. Concatenating feature maps learned y different layers increase variation in the input of susequent layers and improves efficiency.

## DenseNets
Consider a single image $x_0$ passed through a Convolutional network of $L$ layers, each of which implements a non-linear transformation $H_\ell(\cdot)$. ResNets add skip-connection that bypasses the non-linear transformation with an identity function $$ x_\ell = H_\ell(x_{\ell-1}) + x_{\ell - 1}$$ As a reult, the gradients flows directly through the identity function from later layers to the earlier layers. However, the identity function and the output of $H_\ell$ are combined by summation, which may impede the information flow in the network.

To further improve the information flow between layers authors propose a different connectivity pattern:: introducing direct connections from any layer to all subsequent layers. $$ x_\ell = H_\ell\Big([x_0, ..., x_{\ell-1}]\Big)$$ with each $H_\ell$ a composite function of three consecutive operations: batch normalization, RELU, and $3 \times 3$ convolution. To facilitate pooling in the architecture authors divide the network into multiple densely connected dense blocks. The layers between blocks are transition layers with convolution and pooling.

![DenseNet2](http://i.imgur.com/kYTJDTz.png)
> A deep DenseNet with three dense blocks, two layers between two adjacent blocks are referred to as transition layers and change feature map sizes via convolution and pooling

If each function $H_\ell$ produces $k$ feature maps as output, it follows that the $\ell^{th}$ layer has $k \times (\ell - 1) + k_0$ input feature maps, with $k_0$ the number of channels in the input image. To prevent the network from growing too wide and improve parameter efficiency the authors limit $k$ to a small number like 12. It is shown that a relatively small $k$ is sufficient to obtain state-of-the-art results on datasets. Each layer has access to all preceding feature-maps in its block and therefore access to network's "collective knowledge". One can view the feature-maps as the global state of the network with each layer adds $k$ feature maps of its own to tis state. The $k$ regulates how much new information contributes to the global state.

Although each layer produces only $k$ output feature maps, it receives many more inputs. A $1 \times 1$ convolution can be introduced as _bottleneck_ layer before each $3 \times 3$ convolution to reduce the number of input feature maps, thus improving the computational efficiency. The model compactness is further improved by reducing the number of feature-maps at transition layers.




# ADAM: A Method For Stochastic Optimization
![Screen Shot 2017-04-18 at 18.56.51](https://ooo.0o0.ooo/2017/04/19/58f69a176c432.png)
## Algorithm
Let $f(\theta)$ be a noisy objective function, $g_t = \nabla_\theta f_t (\theta)$  the gradient. The algorithm updates exponential moving averges $m_t$ and squared gradient $v_t$ with Hyperparameters $\beta_1, \beta_2$ controling the exponential decay rate. $m_t$ are estimates of the first moment (mean), and $v_t$ are estimates of the second raw moment (uncentered variance) of the gradient. They are initialized to 0 making their bias towards 0.

### Update Rule
The effective step taken in parameter space at timestep $t$ is
$$\Delta_t = \alpha \cdot \frac{\hat{m_t}}{\sqrt{\hat{v_t}}}$$

The effective stepsize has two upper bounds:
At the case of severe sparsity: _when a gradient is 0 at all timesteps except at current step_
$$ m_t = \beta_1 \cdot 0 + (1 - \beta_1)\cdot g_t = (1 - \beta_1)\cdot g_t$$ $$ \hat{m_t} = \frac{1 - \beta_1}{1 - \beta_1^t} g_t $$ $$ likewise, v_t = (1-\beta_2) \cdot g_t^2 $$ $$ \hat{v_t} = \frac{1 - \beta_2}{1 - \beta_2^t} \cdot g_t^2 $$ $$ |\Delta_t| = \alpha \cdot \frac{(1-\beta_1)\sqrt{1-\beta_2^t}}{\sqrt{1-\beta_2}(1-\beta_1^t)}$$ $$\bold{if} \space (1-\beta_1) > \sqrt{1-\beta_2}$$ $$ \Rightarrow  |\Delta_t| \le \alpha \cdot (1-\beta_1) / \sqrt{1-\beta_2}  $$

For less sparse cases, the effective stepsize will be smaller, if $ (1-\beta_1)=\sqrt{1-\beta_2}$ we have that $|\hat{m_t} / \sqrt{\hat{v_t}}| <1 \Rightarrow |\Delta_t| \le \alpha \space $. Usually, since $|\mathbb{E}[g] / \sqrt{\mathbb{E}[g^2]}| < 1$ we have $\hat{m_t} / \sqrt{\hat{v_t}} \approx \pm 1$. The effective magnitude of the steps taken in parameter space at each timestep is approximately bounded by the stepsize setting $\alpha$.

This can be understood as establishing a __trust region__ around the current parameter value. The current gradient estimate does not provide sufficient information beyond this trust region. The effective stepsize is also invariant to the scale of gradients. Let $c \in \mathbb{R}$ the factor that scales $g$. Then we have $\Delta_t = \frac{c\cdot \hat{m_t}}{\sqrt{c^2 \cdot \hat{v_t}}}=\frac{\hat{m_t}}{\sqrt{\hat{v_t}}}$

### Initialization Bias Correction
Here we will derive the term for the second moment estimate, and that for first moment estimate is analogous.

Initialize the exponential moving average $v_0 = \vec{0}$, thus we have
$$v_t = (1-\beta_t)\sum_i^t \beta_2^{t-i}\cdot g_i^2$$
We wish to know how $\mathbb{E}[v_t]$ relates to the true second moment $\mathbb{E}[g_t^2]$, so we can correct the discrepancy between the two.
$$\mathbb{E}[v_t] = \mathbb{E}\Bigg[(1-\beta_2)\sum_i^t \beta_2^{t-i}\cdot g_i^2\Bigg] $$ $$ = \mathbb{E}[g_t^2]\cdot (1-\beta_2)\sum \beta_t^{t-i} + \epsilon$$ $$  = \mathbb{E}[g_t^2]\cdot (1-\beta_2^t) + \epsilon$$

where the term $(1-\beta_t^2)$ is caused by initializing the running average with zeros, and therefore we divide by this term to correct the initializing bias

### Convergence Analysis
Given an arbitrary, unknown sequence of convex functions $f_1(\theta), ..., f_T(\theta)$, the goal to is to predict the parameter $\theta_t$  and evaluate it on a previous unknown cost function $f_t$. Since the nature of sequence is unknown in advance, the algoritgn is evaluated using regret:
$$ R(T) = \sum_t^T [f_t(\theta_t) - f_t(\theta^\star)]$$ $$ \theta^\star = \arg\min_\theta \sum_t^T f_t(\theta)$$the sum of all the previous differences  between the online prediction $f_t(\theta)$ and the best fixed point parameter $f_t(\theta^\star)$ from a feasible set of all previous steps. The Adam is proved to have $O(\sqrt{T})$ regret bound, and is comparable to the best known bound for this general convex online learning algorithm.

**Corollary** _Assume that the function $f_t$ has bounded gradients and distance between any $theta_t$ generated by Adam is bounded. Adam achieves the following guarantee for all $T \ge 1$_
$$ \frac{R(T)}{T} = O(\frac{1}{\sqrt{T}}) $$
$$ \text{thus} \lim_{T\to\infty} \frac{R(T)}{T} = 0$$

### Extension: ADAMAX
In Adam, the update rule for individual weight is to scale their gradients inversely proportional to a scaled $L^2$ norm of their individual current and past gradients. We can generalize the $L^2$ norm based update rule to a $L^p$ norm based update rule. While such variants become numerically unstable for large $p$, in the special case where $p \to \infty$, we have:

Let the stepsize at time $t$ to be inversely proportional to $v_t^{1/p}$
$$ v_t = \beta_2^p v_{t-1} + (1-\beta_2^p) |g_t|^p $$ $$ = (1-\beta_2^p)\sum_i^t \beta_2^{p(t-i)} \cdot |g_i|^p$$
Note that the decay rerm here is equivalently parameterized as $\beta_2^p$ instead of $\beta_2$. Let $p \to \infty$, and let $u_t = lim_{p \to \infty} (v_t)^{1/p}$, then:
$$ u_t = \lim{p \to\infty} (v_t)^{1/p} = \lim_{p \to\infty} \Bigg((1-\beta_2^p)\sum^t \beta_2^{p(t-i)}\cdot |g_i|^p \Bigg)^{1/p}$$ $$ =  \lim_{p \to\infty} (1-\beta_2^p)^{1/p} \Bigg(\sum^t \beta_2^{p(t-i)}\cdot |g_i|^p \Bigg)^{1/p} $$  $$ =  \lim_{p \to\infty}  \Bigg(\sum^t \Big(\beta_2^{(t-i)}\cdot |g_i|\Big)^p \Bigg)^{1/p} $$ $$ = \max(\beta_2^{t-1}\cdot |g_1|,...,\beta_2^1 \cdot |g_t-1|, |g_t| )$$
This corresponds to the remarkably simple recursive formula:
$$ u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$$

# Neural Machine Translation of Rare Words with Subword Units
## Abstract
Previous work addresses the translation of out-of-vocabulary words by backing off to a dictionary. In this paper, the authors introduce a simpler and more effective approachby making the NMT model capable of open-vocabulary translation by encoding are and unknown words as sequences of subword units.

### Byte Pair Coding
BPE is a simple data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single unused byte. The authors adapt this algorithm to merge characters or character sequences.

Initialize the symbol vocabulary with the character vocabulary, and represent each word as a sequence of characters plus a end-of-word symbol. Then iteratively count all symbol pairs and replace each occurrance of most frequent pair with a new symbol which represents a character $n$-gram. Frequent character $n$-grams are merged into a single symbol.

At test time, we first split words into sequence of characters, then apply the learned operations to merge the characters into larger, known symbols. This is applicable to any word, and allows for open-vocabulary network with fixed symbol vocabularies.

There are two methods for applying BPE, learning two independent encodings for source and target vocabulary, or learning the encoding on the union of two vocabularies (joint BPE). The latter improves consistency between the source and target segmentation, e.g. the same name might be segmented differently if apply BPE independently.
