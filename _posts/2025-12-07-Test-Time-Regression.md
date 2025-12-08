---
layout: distill
title: Test Time Regression
description: Associative Recall and connections to Online Learning
tags: linear-RNNs associative-recall online-learning transformers
giscus_comments: true
date: 2025-12-07
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

bibliography: 2025-12-07-Test-Time-Regression.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Problem Setup
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Matrix Memory Modules
  - name: Softmax Attention as Nonparametric Regression
  - name: Online Learning Framework
  - name: Memory Capacity of Memory Modules
  - name: Online Learning for Matrix Memories
  - name: Conclusion and Limitations
  
---

An important task in language modeling is associative recall: to associate a **key** with a **value** and retrieve that value when queried with the key. For example, after processing "The capital of France is Paris," the model should associate the key "capital of France" with the value "Paris." When it needs to recall this fact later, it can query its memory and retrieve the correct information.

Our exposition follows <d-cite key="wang2025testtime"></d-cite>: we will analyze associative recall as a regression problem, re-cast it as an online learning problem, and then examine the memory capacity of memory modules. 

## Problem Setup
In sequence modeling with transformers, we have the following: 
- the input is a sequence of tokens $$x_1,...,x_T$$  and the output is $$y_1,..,y_T$$. 
- we learn embedding vectors for the tokens $$x_1,...,x_T$$. 
- we learn projection matrices $$\mathbf{W}_Q,\mathbf{W}_K, \mathbf{W}_V$$  to map embeddings to query, key, and value vectors. 
The query vector $$\mathbf{q}_{t}$$​ isn't necessarily one of the key vectors; this allows the model to combine information from multiple values. Nevertheless, the idea is that we want to be able to learn associations between tokens and return the relevant token.

 In a transformer's self attention layer, we keep track of the KV cache, which at time $$t$$ is the key value pairs $$S_t=\{(\mathbf{k}_1,\mathbf{v}_1),...,(\mathbf{k}_t,\mathbf{v}_t)\}$$. We then get $$\mathbf{y}_t=\text{SelfAttn}(S_t,\mathbf{q}_t)=\sum_{i=1}^{t} \frac{s(\mathbf{k}_i, \mathbf{q}_t)}{\sum_{j=1}^{t} s(\mathbf{k}_j, \mathbf{q}_t)} \mathbf{v}_i$$ where $$s(\mathbf{k}, \mathbf{q}) = \exp\left(\frac{\mathbf{k}^\top \mathbf{q}}{\sqrt{d_k}}\right)$$.

This can be framed as an associative learning task: Given $$(\mathbf{k}_1,\mathbf{v}_1),...,(\mathbf{k}_t,\mathbf{v}_t)\in \mathbb{R}^{d_k} \times \mathbb{R}^{d_v}$$ we want to calculate some notion of state $$S_t$$ which we use to form a mapping $$m_t=F(S_t).$$ This mapping $$m_t:\mathbb{R}^{d_k} \rightarrow \mathbb{R}^{d_v}$$ should remember the key value pairs $$m_t(\mathbf{k}_i)\approx \mathbf{v}_i$$ and it returns $$\mathbf{\hat{y}}_t=m_t(\mathbf{q}_t)$$. Then we update the state $$S_{t+1}=f(S_{t},(\mathbf{k}_{t+1},\mathbf{v}_{t+1}))$$.

One issue with standard softmax attention is keeping track of the KV cache $$S_t$$, which grows proportionally with sequence length. And performing softmax for time step $$t$$ takes $$O(t)$$ time. What if we can choose $$S_t$$ to take constant memory, and choose $$m_t$$ so that calculating $$m_t(\mathbf{q}_t)$$ takes $$O(1)$$ time?

Our problem consists of 3 parts:
- **Choose the function class $$\mathcal{F}$$ where our memory module $$m_t$$ lives:** here, the state $$S_t$$ can represent weights $$\theta_{t}$$ of $$m_t$$ if $$m_t=f_{\theta_t}$$ is from a parametric function class. Or $$S_t$$ can be key value pairs if $$m_t$$ is from a nonparametric function class.
- **Choose the loss function:** In order to select $$m_{t}$$ from $$\mathcal{F}$$, we need to minimize some objective. Consider the key and value vectors observed so far: $$\mathbf{K}_t=[\mathbf{k}_1,...,\mathbf{k}_t]$$ and $$\mathbf{V}_t=[\mathbf{v}_1,...,\mathbf{v}_t]$$.  We want $$m_t=\text{argmin}_{m\in \mathcal{F}}L(m(\mathbf{K}_t), \mathbf{V}_t)$$ for some loss function $$L$$. We assume the loss decomposes as $$L(m(\mathbf{K}_t), \mathbf{V}_t)=\sum_{i=1}^t c_i\ell(m(\mathbf{k}_i), \mathbf{v}_i)+\lambda R(m)$$.  The function $$\ell$$ is usually $$L_2$$ loss: $$\ell(\mathbf{a},\mathbf{b})=\frac{1}{2}\lVert \mathbf{a}-\mathbf{b} \rVert ^2_2$$. Since $$m_t$$ is being chosen/learned at the forward pass, this is called test time regression.
- **Choose the optimizer:** instead of solving $$\text{argmin}$$ exactly, we can use an approximate algorithm. 

If we pick a parametric function class, then $$S_{t}$$ will take constant memory and will not grow over time. However, computing the loss still requires us to keep track of the sequence of key and value vectors. Later we will consider an online learning approach that allows us to solve for $$m_{t}$$ approximately and have constant memory complexity.

## Matrix Memory Modules
Consider the class of linear functions $$\mathcal{F}=\{m \vert m(\mathbf{k})=\mathbf{Mk}, \ \mathbf{M}\in \mathbb{R}^{d_v \times d_k}\}$$.

For the loss function, let's consider a weighted ridge regression: a weighted sum of squares $$\frac{1}{2} \sum_{i=1}^{t} c_{i}\left \Vert \mathbf{v}_{i} - \mathbf{Mk}_{i}\right \Vert_{2}^{2}$$  with a regularization term $$R(\mathbf{M})=\lambda \left \Vert \mathbf{M}\right \Vert_F.$$ Let $$\mathbf{K}_t\in \mathbb{R}^{t \times d_k }$$ and $$\mathbf{V}_t \in \mathbb{R}^{t \times d_v }$$ be the key and value matrices, and denote $$\mathbf{C}_{t}=\text{diag}(c_{i})\in \mathbb{R}^{t \times t}$$. We have the general formula:

$$
\begin{align*} m_t &= \text{argmin}_{\mathbf{M}\in \mathbb{R}^{d_v \times d_k}} \frac{1}{2} \sum_{i=1}^{t} c_{i}\left \Vert \mathbf{v}_{i} - \mathbf{Mk}_{i}\right \Vert_{2}^{2}+\lambda \left \Vert \mathbf{M}\right \Vert_F^2 \\ &= \text{argmin}_{\mathbf{M}\in \mathbb{R}^{d_v \times d_k}}\frac{1}{2} \left \Vert (\mathbf{V}_t^\top - \mathbf{MK}_t^\top)\mathbf{C}^{1/2}\right \Vert_{F}^{2} +\lambda \left \Vert \mathbf{M}\right \Vert_F^2 \\ &= \begin{cases} {\mathbf{V}_{t}}^{\top}\mathbf{C}_{t} \mathbf{K}_{t} (\mathbf{K}_{t}^{\top}\mathbf{C}_{t} \mathbf{K}_{t}+2\lambda \mathbf{I})^{-1}, & t \geq d_{k} \\ {\mathbf{V}_{t}}^{\top} (\mathbf{K}_{t} {\mathbf{K}_{t}}^{\top}+2\lambda \mathbf{C}_{t})^{-1} \mathbf{K}_{t}, & t < d_{k} \end{cases} \end{align*}
$$

 This is the objective in mesa optimizers <d-cite key="voswald2024uncovering"></d-cite>.

Instead of directly solving for $$m_{t}$$ for each time step, we can actually compute it as a recurrence using the Sherman-Morrison formula: 
- Let $$\mathbf{P}_{t}={\mathbf{V}_{t}}^{\top}\mathbf{C}_{t} \mathbf{K}_{t}$$
- Let $$\mathbf{R}_{t}=(\mathbf{K}_{t}^{\top}\mathbf{C}_{t} \mathbf{K}_{t}+2\lambda \mathbf{I})^{-1}$$
- Update $$\mathbf{P}_{t+1}=\mathbf{P}_{t}+{c_{t+1}}\mathbf{v}_{t+1}\mathbf{k}_{t+1}^\top$$ and $$\mathbf{R}_{t+1} = \mathbf{R}_{t} - \frac{c_{t+1} (\mathbf{R}_{t} \mathbf{k}_{t+1}) (\mathbf{k}_{t+1}^\top \mathbf{R}_{t})}{1 + c_{t+1} \mathbf{k}_{t+1}^\top \mathbf{R}_{t} \mathbf{k}_{t+1}}$$

### Connection to Linear Attention:
From the loss function, drop the regularization term, assume $$\mathbf{C}_{t}=\mathbf{I}$$, and $$\mathbf{K}^\top_t \mathbf{K}_t \approx \mathbf{I}$$. Then we get $$\mathbf{M}_t=\mathbf{V}^\top_t \mathbf{K}_t=\sum_{i=1}^t \mathbf{v}_i \mathbf{k}_i^\top = \mathbf{M}_{t-1}+\mathbf{v}_t\mathbf{k}_t^\top$$. This is the recurrence of the state matrix in linear attention. In this way, linear attention can be thought of as a suboptimal solution to the least squares problem where we assume the key covariance matrix is the identity matrix.

### Connection to Gated Linear Attention
Now consider time decaying weights, $$c_{i}=\gamma_i^{(t)}=\prod_{j=i+1}^t \gamma_i$$ where $$\gamma_i\in[0,1]$$. Let's also drop the regularization term:

$$m_t=\underset{\mathbf{M}\in \mathbb{R}^{d_v \times d_k} }{\text{argmin}}\frac{1}{2} \sum_{i=1}^{t} \gamma_i^{(t)}\left \Vert \mathbf{v}_{i} - \mathbf{Mk}_{i}\right \Vert_{2}^{2}$$

We can apply the previous closed form solution after rescaling the matrices: $$\mathbf{V}_t \mapsto\mathbf{\Gamma}_t^{\frac{1}{2}}\mathbf{V}_t$$ and $$\mathbf{K}_t \mapsto\mathbf{\Gamma}_t^{\frac{1}{2}}\mathbf{K}_t$$ where $$\mathbf{\Gamma}_t\in\mathbb{R}^{t\times t}$$ is the diagonal matrix with $$(\mathbf{\Gamma}_t)_{ii}=\gamma_i^{(t)}$$. The solution becomes

$$m_t=
\begin{cases} {\mathbf{V}_{t}}^{\top}\mathbf{\Gamma}_t \mathbf{K}_{t} (\mathbf{K}_{t}^{\top}\mathbf{\Gamma}_t \mathbf{K}_{t})^{-1}, & t \geq d_{k} \\ {\mathbf{V}_{t}}^{\top} (\mathbf{K}_{t} {\mathbf{K}_{t}}^{\top})^{-1} \mathbf{K}_{t}, & t < d_{k} \end{cases}$$

If we further assume $$\mathbf{K}_{t}^{\top}\mathbf{\Gamma}_t \mathbf{K}_{t}\approx \mathbf{I}$$, we get 

$$\mathbf{M}_t\approx\sum_{i=1}^{t} \gamma^{(t)}_i \mathbf{v}_i\mathbf{k}_i^\top=\gamma^t\sum_{i=1}^{t-1} \gamma^{(t-1)}_i \mathbf{v}_i\mathbf{k}_i^\top+\mathbf{v}_t\mathbf{k}_t^\top=\gamma^t\mathbf{M}_{t-1}+\mathbf{v}_t\mathbf{k}_t^\top$$

this is the recurrence for gated linear attention, and it shows how it is a suboptimal solution to weighted least squares.

## Softmax Attention as Nonparametric Regression
Here we generalize the notion of inner product to a higher dimensional kernel space. Let $$k:\mathbb{R}^{d_{k}\times d_{k}}\rightarrow \mathbb{R}$$ be a continuous symmetric positive-definite kernel. Mercer's theorem allows us to express the kernel as $$k(\mathbf{x},\mathbf{y})=\phi(\mathbf{x})^\top \phi(\mathbf{y})$$ for a possibly infinite dimensional feature map $$\phi: \mathbb{R}^{d_{k}}\to \mathbb{R}^{d_{\phi}}.$$

Suppose we want to solve $$\mathbf{m}_{t}=\text{argmin}_{\mathbf{m}\in \mathcal{F}} \sum_{i=1}^t \lVert \mathbf{v}_{i}-\mathbf{M}\phi(\mathbf{k}_{i}) \rVert^2_{2}$$ where $$\mathcal{F}=\{ m \vert {m}(\mathbf{k})=\mathbf{M}\phi(\mathbf{k}), \mathbf{M}\in \mathbb{R}^{d_{v}\times d_{\phi}} \}$$. The solution is given by 

$${m}_{t}(\mathbf{q})=\mathbf{V}_{t}^\top \mathbf{k}(\mathbf{K}_{t},\mathbf{K}_{t})^{-1} \mathbf{k}(\mathbf{K}_{t},\mathbf{q}) $$

where $${k}(\mathbf{K}_{t},\mathbf{K}_{t})_{ij}=k(\mathbf{k}_{i},\mathbf{k}_{j})$$. Here we are assuming that $$t\leq d_{\phi}$$. If we choose a polynomial kernel, then $${m}_{t}$$ can be expressed as a $$d_{v} \times d_{\phi}$$ matrix $$\mathbf{M}$$. On the other hand, if we choose an infinite dimensional kernel, then the state becomes the $$t \times t$$ gram matrix $${k}(\mathbf{K}_{t},\mathbf{K}_{t})$$, which increases with sequence length $$t$$. 
Suppose we use the exponential kernel $$k(\mathbf{k}_{i},\mathbf{k}_{j})=\exp (\mathbf{k}_{i}^\top \mathbf{k}_{j}/\sqrt{ d_{k} })$$ and assume $${k}(\mathbf{K}_t,\mathbf{K}_{t})\approx \mathbf{I}$$. Then $${m}_{t}(\mathbf{q})\approx \mathbf{V}^\top_{t}{k}(\mathbf{K}_{t},\mathbf{q})  =\sum_{i=1}^t \mathbf{v}_{i}k(\mathbf{k}_{i}, \mathbf{q})=\sum_{i=1}^t \mathbf{v}_{i} \exp\left( \frac{\mathbf{k}^\top_{i}\mathbf{q}}{\sqrt{ d_{k} }} \right)$$. This is unnormalized softmax, and can be considered a suboptimal solution to the nonparametric regression problem.

## Online Learning Framework
In the previous sections, solved the least squares problem exactly for each time step. Here we compute approximate solutions by casting the problem into an online learning framework. This provides a lot of flexibility in choosing the loss and the optimizer.

Online learning is a sequential decision making problem where a learner makes predictions at each round, with the goal of minimizing the cumulative loss <d-cite key="Shalev-Shwartz_2011"></d-cite>. We have the following problem setup: 
- at each time $$t$$, pick an action $$a_t\in \mathcal{A}$$ from a convex set $$\mathcal{A}$$
- observe the convex loss function $$\ell_t:\mathcal{A}\rightarrow \mathbb{R}$$
- receive loss $$\ell_t(a_t)$$

The goal is to minimize the regret by time horizon $$T$$, which is 
$$\text{regret}_T=\sum_{t=1}^T\ell_t(a_t)-\min_{a\in \mathcal{A}}\sum_{t=1}^T\ell_t(a)$$

We can now interpret test time learning as an online learning problem:
- at time $$t$$, pick a function $$m_{t-1}\in \mathcal{F}$$ (to be elaborated below)
- observe $$(\mathbf{k}_t,\mathbf{v}_t)$$, which produces a loss function $$\ell_t(m)=\left \Vert m(\mathbf{k}_t)-\mathbf{v}_t \right \Vert_2^2$$
- receive loss $$\ell_t(m_{t-1})$$
- in the next round, choose $$m_{t}$$. For example, we can take a gradient descent step $$m_t=m_{t-1}-\eta_t \nabla \ell_t(m_{t-1})$$. In other scenarios, we choose $$m_{t}=\text{argmin}_{m\in\mathcal{F}}\ell_{t}(m)$$. 
- we want to minimize regret by time $$T$$, which is the end of the sequence. In other words:
$$ \text{regret}_{T}=\sum_{t=1}^T \left \Vert m_{t-1}(\mathbf{k}_t)-\mathbf{v}_t \right \Vert_2^2-\min_{m\in \mathcal{F}}\sum_{t=1}^T \left \Vert m(\mathbf{k}_t)-\mathbf{v}_t \right \Vert_2^2$$.

The benefit of the online learning approach is that we have natural way to expand on this:
- changing the function class to be neural networks <d-cite key="sun2025learning"></d-cite>.
- modifying the loss function to be least squares of a set of points <d-cite key="behrouz2025atlas"></d-cite>.
- choose another optimizer.

Here are some examples:

<div class="table-responsive" style="overflow-x: auto; -webkit-overflow-scrolling: touch;">

<table class="table table-sm">
  <thead>
    <tr>
      <th>Method</th>
      <th>State</th>
      <th>online loss $\ell_t(m)$</th>
      <th>optimizer</th>
      <th>state update rule</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear Attention <d-cite key="katharopoulos2020transformers"></d-cite></td>
      <td>$\mathbf{M}_{t}\in \mathbb{R}^{d_{v}\times d_{k}}$</td>
      <td>$-\langle \mathbf{Mk}_{t},\mathbf{v}_{t} \rangle$</td>
      <td>GD ($\eta_{t}=1$)</td>
      <td>$\mathbf{M}_{t}=\mathbf{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^\top$</td>
    </tr>
    <tr>
      <td>RetNet <d-cite key="sun2023retentive"></d-cite></td>
      <td>$\mathbf{M}_{t}\in \mathbb{R}^{d_{v}\times d_{k}}$</td>
      <td>$-\langle \mathbf{Mk}_{t},\mathbf{v}_{t} \rangle+\frac{1-\alpha}{2}\lVert \mathbf{M} \rVert^2_{F}$</td>
      <td>GD ($\eta_{t}=1$)</td>
      <td>$\mathbf{M}_{t}=\alpha\mathbf{M}_{t-1}+\mathbf{v}_{t}\mathbf{k}_{t}^\top$</td>
    </tr>
    <tr>
      <td>DeltaNet <d-cite key="yang2025parallelizing"></d-cite></td>
      <td>$\mathbf{M}_{t}\in \mathbb{R}^{d_{v}\times d_{k}}$</td>
      <td>$\frac{1}{2} \left \Vert \mathbf{M} \mathbf{k}_t-\mathbf{v}_t \right \Vert^2_2$</td>
      <td>GD</td>
      <td>$\mathbf{M}_t=\mathbf{M}_{t-1}(1-\eta_t\mathbf{k}_{t}\mathbf{k}_{t}^\top)+\eta_t\mathbf{v}_t\mathbf{k}_t^\top$</td>
    </tr>
    <tr>
      <td>Regularized DeltaNet</td>
      <td>$\mathbf{M}_{t}\in \mathbb{R}^{d_{v}\times d_{k}}$</td>
      <td>$\frac{1}{2} \left \Vert \mathbf{M} \mathbf{k}_t-\mathbf{v}_t \right \Vert^2_2+\frac{\lambda_{t}}{2}\lVert \mathbf{M} \rVert^2_{F}$</td>
      <td>GD</td>
      <td>$\mathbf{M}_t=\mathbf{M}_{t-1}((1-\eta_{t}\lambda_{t})-\eta_t\mathbf{k}_{t}\mathbf{k}_{t}^\top)+\eta_t\mathbf{v}_t\mathbf{k}_t^\top$</td>
    </tr>
    <tr>
      <td>Gated DeltaNet <d-cite key="yang2025gated"></d-cite></td>
      <td>$\mathbf{M}_{t}\in \mathbb{R}^{d_{v}\times d_{k}}$</td>
      <td>$\lVert M-\alpha_{t}M_{t-1} \rVert^2_{F}-2\langle Mk_{t},\beta_{t}(v_{t}-\alpha_{t}M_{t-1}k_{t}) \rangle$</td>
      <td>exact</td>
      <td>$\mathbf{M}_t=\mathbf{M}_{t-1}(\alpha_{t}(1-\beta_t\mathbf{k}_{t}\mathbf{k}_{t}^\top))+\beta_t\mathbf{v}_t\mathbf{k}_t^\top$</td>
    </tr>
    <tr>
      <td>TTT MLP <d-cite key="sun2025learning"></d-cite></td>
      <td>parameters $\theta_{t}$ of neural network $f$</td>
      <td>$\frac{1}{2} \left \Vert f_{\theta}(\mathbf{k}_t)-\mathbf{v}_t \right \Vert^2_2$</td>
      <td>GD (fixed $\eta$)</td>
      <td>$\theta_{t}=\theta_{t-1}-\eta \nabla L_{t}(\theta_{t-1})$</td>
    </tr>
    <tr>
      <td>Titans <d-cite key="behrouz2024titans"></d-cite></td>
      <td>parameters $\theta_{t}$</td>
      <td>$\frac{1}{2} \left \Vert f_{\theta}(\mathbf{k}_t)-\mathbf{v}_t \right \Vert^2_2$</td>
      <td>GD w/ momentum\* (w/ weight decay)</td>
      <td>$\theta_{t}=\alpha_{t}\theta_{t-1}+\beta_{t}$<br>$\beta_{t}=\mu_{t}\beta_{t-1}-\eta_{t} \nabla L_{t}(\theta_{t-1})$</td>
    </tr>
    <tr>
      <td>Atlas <d-cite key="behrouz2025atlas"></d-cite></td>
      <td>parameters $\theta_{t}$</td>
      <td>$\sum_{i=t-c+1}^t\frac{\gamma_{i}}{2} \left \Vert f_{\theta}(\mathbf{k}_i)-\mathbf{v}_i \right \Vert^2_2$</td>
      <td>Muon (w/ weight decay)</td>
      <td>$\theta_{t}=\alpha_{t}\theta_{t-1}-\eta_{t}\text{NS}(\beta_{t})$  (Newton -Schultz) $\beta_{t}=\mu_{t}\beta_{t-1}+ \nabla L_{t}(\theta_{t-1})$</td>
    </tr>
  </tbody>
</table>

</div>

Note that for Titans and Atlas we have a decoupled weight decay, in the style of AdamW. Also, the momentum formulation in Titans is slightly different in that the learning rate $$\eta_{t}$$ is on the gradient and not the momentum term $$\beta_{t}$$.

## Memory Capacity of Memory Modules
A natural question to ask is how many key value pairs can a memory module remember. Suppose the key and value dimension are equal to $$d$$. If we have $$d$$ pairs of vectors $$(\mathbf{k}_{i},\mathbf{v}_{i})$$ such that $$\mathbf{k}_{i}$$ are linearly independent, we can always find an $$d\times d$$ matrix $$\mathbf{M}$$ such that $$\mathbf{v}_{i}=\mathbf{Mk}_{i}$$. However, in language modeling, we don't have to return the value vector exactly. Instead, we want to choose the token associated with the value vector. Suppose we have $$N$$ key-value pairs and we use "hard-max" attention: we want a vector $$\hat{\mathbf{v}}=\mathbf{Mk}_{i}$$ that maximizes a score function $$s$$ so that $$i=\underset{j\in[N]}{\text{argmax}}\ s(\hat{\mathbf{v}},\mathbf{v}_{j}).$$ In this section we will consider the inner product as the scoring function: $$s(\hat{\mathbf{v}},\mathbf{v}_{j})=\hat{\mathbf{v}}^\top \mathbf{v}_{j}$$.

It turns out that the number of key value pairs a matrix memory module can retain is proportional to the number of parameters in the matrix (up to logarithmic factors). 

Here we introduce some results from <d-cite key="nichani2024understanding"></d-cite>.
Suppose $$d=d_{v}=d_{k}$$ and consider a set of input tokens $$[N]$$ and a set of output tokens $$[M]$$. We want to learn an association, given by $$f^*:[N]\to [M]$$. Assume that $$f^*(i)=i$$. For each $$x\in[N]$$, assign an embedding vector $$\mathbf{k}_{x}\in \mathbb{R}^d$$ and for each $$y\in[M]$$ assign an unembedding vector $$\mathbf{v}_{y}\in \mathbb{R}^d$$, where both vectors lie on the unit sphere: $$\mathbf{k}_{x}, \mathbf{v}_{y}\in \mathbb{S}^{d-1}$$. Let $$m:\mathbb{R}^d \to \mathbb{R}^d$$ be our memory module, which will learn key value associations, $$m(\mathbf{k}_{i}) \approx\mathbf{v}_{i}$$. Our prediction is given by argmax decoding $$\hat{f}(x):=\text{argmax}_{y\in [M]}\mathbf{v}_{y}^\top m(\mathbf{k}_{x})$$. If $$m$$ is a linear map $$m(\mathbf{k}_{x})=\mathbf{M}\mathbf{k}_{x}$$, we have the following result:
#### Theorem 1 <d-cite key="nichani2024understanding"></d-cite>: 
Assume that $$f^*$$ is injective. If $$d^2 \gtrsim N \cdot \text{poly log} N$$, then with high probability over the draw of the embeddings, there exists a $$\mathbf{W}$$ such that
$$
\arg\max_{y \in [M]} \mathbf{u}_y^\top \mathbf{We}_x = f^*(x) \quad \text{for all } x \in [N].
$$
This capacity is obtained by the construction $$W = \sum_{x \in [N]} \mathbf{u}_{f^*(x)} \mathbf{e}_x^\top$$. 

Now consider a 2 layer neural network with hidden dimension $$c$$, given by $$m(\mathbf{k})=\mathbf{V}\sigma(\mathbf{Wk})$$ for matrices $$\mathbf{V},\mathbf{W}\in \mathbb{R}^{c\times d}$$. At first, we might think that the greater expressivity of neural networks can translate to a greater number of key-value pairs remembered. However, we get a similar result as the linear case:
#### Theorem 2 <d-cite key="nichani2024understanding"></d-cite>:
If $$cd \gtrsim N \cdot \text{poly log} N$$, then with high probability over the draw of the embeddings, there exists $$V, W$$ such that
$$
\arg\max_{y \in [M]} \mathbf{u}_y^\top \mathbf{V}^\top \sigma(\mathbf{We}_x) = f^*(x) \quad \text{for all } x \in [N].
$$
Note that both linear and nonlinear memory modules have a capacity that is proportional to the number of parameters (ignoring log factors). This suggests that in terms of capacity alone, nonlinear memory modules don't have an advantage. However, with nonlinear modules, $$f^*$$ does not have to be injective, which means several key vectors can be associated to a single value vector.

## Online Learning for Matrix Memories
Now let's look at numerical experiments to see how different matrix memory modules fare. You can find the code [here.](https://github.com/stephan271c/in-context-recall) We compare 3 modules:
- Linear Attention
- DeltaNet (denoted TTT)
- Mesa Layer 
For the TTT module, instead of using only the online loss $$L_{t}(\mathbf{M})= \left \Vert \mathbf{M} \mathbf{k}_t-\mathbf{v}_t \right \Vert^2_2$$, we introduce a context size parameter $$\text{ctx}$$ so that the loss becomes $$L_{t}(\mathbf{M})=\sum_{i=0}^{\text{ctx}-1}\left \Vert \mathbf{M} \mathbf{k}_t-\mathbf{v}_t \right \Vert^2_2$$. 

For the data, we sample $$20$$-dimensional key and value vectors of unit length. By default, each vector is drawn i.i.d, but later we change the pairwise correlation of the key vectors, so that $$\text{corr}(k_{i},k_{i-1})\neq 0$$. We sample 500 sequences of length 100 and plot the average accuracy and recall count. 

### Evaluation metrics
We consider two metrics: accuracy by offset, and retrieval counts by timestep.
#### Accuracy by offset
After updating our memory module $$\mathbf{M}_{t}=\text{update}(\mathbf{M}_{t-1},(\mathbf{k}_{t},\mathbf{v}_{t}))$$, we test whether it can retrieve the value vector $$\mathbf{v}_{i}$$ when given the key vector $$\mathbf{k}_{i}$$. In other words, we compute $$\frac{1}{\text{# of seq}}\sum_{s=1}^{\text{# of seq}}\mathbf{1}_{i=r_{i}}$$, where  $$r_{i}=\text{argmax}_{j\in[0,t]}\mathbf{v}^\top_{j}(\mathbf{M}_{t}\mathbf{k}_{i})$$. Note that we compare inner product scores of only the value vectors seen so far, $$\mathbf{v}_{0},\dots \mathbf{v}_{t}$$. This is analogous to causal language modeling with transformers, where we compute the attention scores against previously seen tokens. 
We then organize these retrieval accuracies by offset: we look at how well $$\mathbf{M}_{t}$$ retrieves $$\mathbf{v}_{t-k}$$ for fixed $$k$$, as $$t$$ varies from $$1$$ to the sequence length. Note that the accuracy calculation involves a decreasing number of subsequences as the offset increases; at offset $$k$$, we can only include retrieval counts when $$t\geq k$$.

#### Retrieval counts by timestep
Here we sum up all the correct retrievals of $$\mathbf{M}_{t}$$ for each timestep $$t$$. If the memory module is able to retrieve all previously seen key value pairs, the graph should look like the line $$y=x$$.

### Empirical results

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/2025-12-07-Test-Time-Regression/corr0-ctx1-acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

In this graph, we plot recall accuracy by offset, with a context size of 1. Note that linear attention and Mesa Layer attention are independent of the context size. 

Due to the exponential forgetting parameter $$\gamma = 0.9$$, the Mesa Layer perfectly recalls the most recent key value pairs. On the other hand, linear attention has a nonzero retrieval accuracy across all offsets. This echos the observation that linear attention lacks a forgetting mechanism, so it cannot properly attend to new tokens once its capacity is saturated. The TTT layer does seem to have a forgetting mechanism, with a higher accuracy for more recent KV pairs. Additionally, a larger learning rate corresponds to faster learning and faster decay of recall accuracy.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/2025-12-07-Test-Time-Regression/corr0-ctx5-acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Here we examine the TTT layer with context size 5. This effectively means that the matrix memory can be updated 5 times for a single KV pair through gradient descent. Note that the accuracy at the first few offsets is low, and it reaches perfect accuracy faster when using a larger learning rate. It seems that a single gradient step isn't enough for the TTT layer.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/2025-12-07-Test-Time-Regression/corr0.7-ctx5-acc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Here we introduce pairwise correlation between the key vectors, so that $$\text{corr}(\mathbf{k}_{i},\mathbf{k}_{i-1})=0.7$$. Both linear attention and the TTT layer suffer in performance. This makes sense, as neither memory module incorporates the covariance matrix $$\mathbf{K}_{t}^\top\mathbf{K}_{t}$$ into its update equation. On the other hand, the mesa layer maintains good performance and retains the same shape curve.

<div class="row mt-3">
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/2025-12-07-Test-Time-Regression/corr0-ctx1-ret.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/2025-12-07-Test-Time-Regression/corr0-ctx5-ret.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/2025-12-07-Test-Time-Regression/corr0.7-ctx5-ret.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Here we see the retrieval counts by timestep. For all memory modules, the retrieval count saturates at much lower numbers than what is theoretically possible, which should be roughly proportional to the number of parameters in the matrix. However, in the 0 input correlation case, the counts saturate at above 20 counts, which is more than the maximum rank of our $$20\times 20$$ matrix.
Note that for the TTT layers and linear attention, the retrieval count starts to decrease as the timestep increases. It's likely that even the TTT layers aren't fully forgetting past context, negatively affecting retrieval capacity later in the sequence. On the other hand, the retrieval count remains stable for the Mesa Layer.

## Conclusion and Limitations
We gave an overview of associative recall and how it can be framed as an online learning problem. We also looked at a toy example of doing associative recall on key value pairs of vectors on the unit sphere. A few things we didn't cover are more advanced optimizers, as well as multi-layer TTT. They didn't do as well in preliminary experiments, likely because the vectors are sampled randomly for each sequence, so there's less structure to exploit. In language modeling, we learn the embedding matrices as well as the projection matrices, so it's possible for more sophisticated memory modules to perform better. We also didn't consider computational efficiency- often times a paper will introduce a memory module and then implement it using an approximation.
