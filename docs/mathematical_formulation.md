# Mathematical Formulation of VectorFin

This document provides a formal mathematical description of the VectorFin financial prediction system, detailing the key components and their interactions.

## 1. Notation

Let us define:

- $\mathbf{X}_\text{text} = \{x_1, x_2, ..., x_n\}$: A set of $n$ financial text documents (news headlines, reports, etc.)
- $\mathbf{X}_\text{market} = \{m_1, m_2, ..., m_k\}$: Market data for $k$ tickers over time
- $\mathbf{v}_\text{text} \in \mathbb{R}^d$: Text vector in shared embedding space (dimension $d$)
- $\mathbf{v}_\text{market} \in \mathbb{R}^d$: Market data vector in shared embedding space (dimension $d$)
- $\mathbf{u} \in \mathbb{R}^f$: Unified representation vector (dimension $f$)
- $y_\text{direction} \in [0, 1]$: Prediction of price movement direction (probability of increase)
- $y_\text{magnitude} \in \mathbb{R}$: Prediction of price movement magnitude (percentage)
- $y_\text{volatility} \in \mathbb{R}^+$: Prediction of price volatility

## 2. Text Vectorization

The transformation of financial text into vector representations occurs through:

$$\mathbf{v}_\text{text} = \Phi_\text{text}(\mathbf{X}_\text{text})$$

Where $\Phi_\text{text}$ is a composite function:

$$\Phi_\text{text} = \mathcal{P} \circ \mathcal{C} \circ \mathcal{T} \circ \mathcal{S}$$

With:

- $\mathcal{S}$: Text preprocessing and standardization
- $\mathcal{T}$: Transformer encoder (e.g., FinBERT) mapping text to contextual embeddings
- $\mathcal{C}$: Concatenation with sentiment features
- $\mathcal{P}$: Projection to shared vector space

For a single text document $x_i$, the operations are:

1. **Preprocessing**: $x_i' = \mathcal{S}(x_i)$
2. **Transformer Encoding**: $\mathbf{h}_i = \mathcal{T}(x_i')$ where $\mathbf{h}_i \in \mathbb{R}^{d_T}$
3. **Sentiment Feature Extraction**: $\mathbf{s}_i = \mathcal{W}_s \mathbf{h}_i$ where $\mathbf{s}_i \in \mathbb{R}^{d_s}$
4. **Concatenation**: $\mathbf{c}_i = [\mathbf{h}_i; \mathbf{s}_i]$ where $\mathbf{c}_i \in \mathbb{R}^{d_T + d_s}$
5. **Projection**: $\mathbf{v}_{\text{text},i} = \mathcal{P}(\mathbf{c}_i) = \mathcal{W}_p\mathbf{c}_i + \mathbf{b}_p$ where $\mathbf{v}_{\text{text},i} \in \mathbb{R}^d$

$\mathcal{W}_s$ and $\mathcal{W}_p$ are learnable parameter matrices, and $\mathbf{b}_p$ is a bias vector.

## 3. Market Data Vectorization

The market data is transformed into the shared vector space through:

$$\mathbf{v}_\text{market} = \Phi_\text{market}(\mathbf{X}_\text{market})$$

Where $\Phi_\text{market}$ is a composite function:

$$\Phi_\text{market} = \mathcal{E} \circ \mathcal{N} \circ \mathcal{F}$$

With:

- $\mathcal{F}$: Feature engineering from raw market data
- $\mathcal{N}$: Normalization of features
- $\mathcal{E}$: Embedding into shared vector space

For market data $m_j$, the operations are:

1. **Feature Engineering**: $\mathbf{f}_j = \mathcal{F}(m_j)$ where $\mathbf{f}_j \in \mathbb{R}^{d_f}$ includes:

   - Price indicators (open, high, low, close)
   - Volume indicators
   - Technical indicators (moving averages, RSI, MACD, etc.)
   - Volatility measures

2. **Normalization**: $\mathbf{f}_j' = \mathcal{N}(\mathbf{f}_j) = \frac{\mathbf{f}_j - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$ where $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are mean and standard deviation vectors

3. **Embedding**: $\mathbf{v}_{\text{market},j} = \mathcal{E}(\mathbf{f}_j') = \mathcal{W}_e\mathbf{f}_j' + \mathbf{b}_e$ where $\mathbf{v}_{\text{market},j} \in \mathbb{R}^d$

$\mathcal{W}_e$ and $\mathbf{b}_e$ are learnable parameters.

## 4. Alignment and Integration

The alignment between text and market vectors is achieved through a cross-modal attention mechanism:

$$\mathbf{u} = \mathcal{A}(\mathbf{v}_\text{text}, \mathbf{v}_\text{market})$$

Where $\mathcal{A}$ is the alignment function implemented as:

$$\mathcal{A}(\mathbf{v}_\text{text}, \mathbf{v}_\text{market}) = \mathcal{G}(\mathbf{v}_\text{text}^{\text{attn}}, \mathbf{v}_\text{market}^{\text{attn}})$$

The attention is computed as:

1. **Query, Key, Value Projections**:

   - For text vectors: $\mathbf{Q}_t = \mathbf{W}_{qt}\mathbf{v}_\text{text}$, $\mathbf{K}_t = \mathbf{W}_{kt}\mathbf{v}_\text{text}$, $\mathbf{V}_t = \mathbf{W}_{vt}\mathbf{v}_\text{text}$
   - For market vectors: $\mathbf{Q}_m = \mathbf{W}_{qm}\mathbf{v}_\text{market}$, $\mathbf{K}_m = \mathbf{W}_{km}\mathbf{v}_\text{market}$, $\mathbf{V}_m = \mathbf{W}_{vm}\mathbf{v}_\text{market}$

2. **Cross-Modal Attention Scores**:

   - Text attending to market: $\mathbf{A}_{tm} = \text{softmax}\left(\frac{\mathbf{Q}_t \mathbf{K}_m^T}{\sqrt{d_k}}\right)$
   - Market attending to text: $\mathbf{A}_{mt} = \text{softmax}\left(\frac{\mathbf{Q}_m \mathbf{K}_t^T}{\sqrt{d_k}}\right)$

3. **Context Vectors**:

   - Text context: $\mathbf{v}_\text{text}^{\text{attn}} = \mathbf{A}_{tm}\mathbf{V}_m$
   - Market context: $\mathbf{v}_\text{market}^{\text{attn}} = \mathbf{A}_{mt}\mathbf{V}_t$

4. **Fusion Function** $\mathcal{G}$:
   - $\mathbf{u} = \mathcal{G}(\mathbf{v}_\text{text}^{\text{attn}}, \mathbf{v}_\text{market}^{\text{attn}}) = \text{LayerNorm}\left(\mathcal{W}_g[\mathbf{v}_\text{text}^{\text{attn}}; \mathbf{v}_\text{market}^{\text{attn}}] + \mathbf{b}_g\right)$

Where $d_k$ is the dimensionality of the keys, and $\mathcal{W}_g$, $\mathbf{b}_g$ are learnable parameters.

## 5. Prediction Heads

From the unified representation $\mathbf{u}$, the model makes predictions through specialized heads:

### 5.1 Direction Prediction

$$y_\text{direction} = \sigma\left(\mathcal{H}_\text{direction}(\mathbf{u})\right)$$

Where $\sigma$ is the sigmoid activation function, and $\mathcal{H}_\text{direction}$ is a multi-layer perceptron:

$$\mathcal{H}_\text{direction}(\mathbf{u}) = \mathbf{W}_{d2}\text{ReLU}(\mathbf{W}_{d1}\mathbf{u} + \mathbf{b}_{d1}) + \mathbf{b}_{d2}$$

### 5.2 Magnitude Prediction

$$y_\text{magnitude} = \mathcal{H}_\text{magnitude}(\mathbf{u})$$

Where $\mathcal{H}_\text{magnitude}$ is a multi-layer perceptron:

$$\mathcal{H}_\text{magnitude}(\mathbf{u}) = \mathbf{W}_{m2}\text{ReLU}(\mathbf{W}_{m1}\mathbf{u} + \mathbf{b}_{m1}) + \mathbf{b}_{m2}$$

### 5.3 Volatility Prediction

$$y_\text{volatility} = \text{ReLU}\left(\mathcal{H}_\text{volatility}(\mathbf{u})\right)$$

Where $\mathcal{H}_\text{volatility}$ is a multi-layer perceptron, and ReLU ensures non-negative output:

$$\mathcal{H}_\text{volatility}(\mathbf{u}) = \mathbf{W}_{v2}\text{ReLU}(\mathbf{W}_{v1}\mathbf{u} + \mathbf{b}_{v1}) + \mathbf{b}_{v2}$$

## 6. Loss Functions and Training

The model is trained with multiple loss components:

1. **Direction Loss**: Binary cross-entropy
   $$\mathcal{L}_\text{direction} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

   Where $y_i$ is the true direction label (1 for increase, 0 for decrease), and $p_i$ is the predicted probability of increase.

2. **Magnitude Loss**: Mean squared error
   $$\mathcal{L}_\text{magnitude} = \frac{1}{N}\sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2$$

   Where $y_i$ is the true percentage change, and $\hat{y}_i$ is the predicted change.

3. **Volatility Loss**: Mean squared error
   $$\mathcal{L}_\text{volatility} = \frac{1}{N}\sum_{i=1}^{N} \left( v_i - \hat{v}_i \right)^2$$

   Where $v_i$ is the true volatility, and $\hat{v}_i$ is the predicted volatility.

4. **Contrastive Loss**: For aligning text and market representations
   $$\mathcal{L}_\text{contrastive} = \frac{1}{N}\sum_{i=1}^{N} \left[ d(\mathbf{v}_{\text{text},i}, \mathbf{v}_{\text{market},i}) - d(\mathbf{v}_{\text{text},i}, \mathbf{v}_{\text{market},j}) + \alpha \right]_+$$

   Where $d(\cdot,\cdot)$ is a distance function, $\mathbf{v}_{\text{market},j}$ is a negative sample, $\alpha$ is a margin parameter, and $[\cdot]_+$ denotes $\max(0, \cdot)$.

The total loss is a weighted combination:

$$\mathcal{L}_\text{total} = \lambda_1 \mathcal{L}_\text{direction} + \lambda_2 \mathcal{L}_\text{magnitude} + \lambda_3 \mathcal{L}_\text{volatility} + \lambda_4 \mathcal{L}_\text{contrastive}$$

Where $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ are weight coefficients that balance the contribution of each loss component.

## 7. Interpretation Layer

For model interpretability, attention weights can be extracted to analyze the contribution of different features:

1. **Text contribution**: $\mathbf{c}_\text{text} = \mathbf{A}_{tm} \cdot \mathbf{1}$
2. **Market feature contribution**: $\mathbf{c}_\text{market} = \mathbf{A}_{mt} \cdot \mathbf{1}$

Where $\mathbf{1}$ is a vector of ones of appropriate dimension.

These contributions can be visualized to provide insights into which text segments or market features most influenced the prediction.
