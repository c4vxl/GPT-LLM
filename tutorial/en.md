# How a GPT-LLM Works

## Table of contents
1. [Tokenization](#1-tokenization)
2. [Embedding](#2-embedding)
   1. [Token Embedding](#21-token-embedding)
   2. [Position Embedding](#22-position-embedding)
3. [Transformer Blocks](#3-transformer-block)
   1. [Attention](#1-attention)
      1. [Single Head Attention](#11-single-head-attention)
      2. [Multi Head Attention](#12-multi-head-attention)
   2. [Feed-Forward](#2-feed-forward)
   3. [Layer Normalization](#3-layer-normalization)
   4. [Connections and Residual Connections](#4-connections-and-residual-connections)

---

### 1. Tokenization
Tokenization is a process that occurs outside the model. Here, the input (in the form of words) is converted into so-called `tokens`. A token is a number representing a word (`word tokenization`), a part of a word (`sub-word tokenization`), or a single character (`character tokenization`).

### 2. Embedding
An embedding is essentially a mapping of discrete objects (in this case, tokens) to vectors of real numbers in a continuous space.

During model initialization, both token and position embeddings are created (`wte` and `wpe`).

#### 2.1 Token Embedding:
Token Embedding (`wte`) carries information about what each individual token in the input sequence means in the context of the other tokens.

#### 2.2 Position Embedding:
Since only one token can be processed by the model at a time, the model needs to know the position of the current token in the input sequence. This is where position embedding (`wpe`) comes into play. Position embedding is an embedding in which the position of each token in the input sequence is encoded.

### 3. Transformer Block
Transformer blocks are fundamental building blocks in Transformer architectures used to process input data and generate output representations. The Transformer block consists of several layers of submodules that work together to transform the input data.

<details>
    <summary>1. Attention</summary>

#### 1. Attention

Attention is a mechanism that allows a model to focus on relevant parts of the input while generating an output. It works similar to human attention, which focuses on different parts of a sentence or scene to understand or respond to it.

##### 1.1 "Single-Head Attention":
In `Single-Head Attention`, a single "head" calculates attention weights between different tokens in the input. For each token, a "query," "key," and "value" are generated. The "query" represents the current position, while the "keys" represent the other positions in the input. The similarity between the "query" and the "keys" is calculated to determine the attention weights, indicating how much attention is given to each token with respect to the others. The "values" represent the values to be weighted, which are then combined with the calculated weights to produce the output.

##### 1.2 "Multi-Head Attention":
In `Multi-Head Attention`, multiple `Single-Heads` operate in parallel. Each Single-Head learns to capture different types of attention, allowing the model to consider various aspects of the input. Subsequently, the outputs of individual attention heads are combined to obtain a more comprehensive representation of attention.
</details>

<details>
    <summary>2. Feed-Forward</summary>

#### 2. Feed-Forward
`Feed Forward Layers` enable the model to capture complex nonlinear relationships between different parts of the input data and learn richer representations of the data.

Feed-Forward layers typically consist of two linear transformations:
1. A linear transformation that maps the input data to a higher-dimensional space.
2. Another linear transformation that reduces the dimensionality back to the original dimension.

Between these linear transformations, a non-linear activation function such as `ReLU (Rectified Linear Unit)` is typically applied to capture nonlinear relationships in the data and increase the model's expressiveness.
</details>

<details>
    <summary>3. Layer Normalization</summary>

#### 3. Layer Normalization
Layer Normalization is a technique used to improve training stability and increase convergence speed. Layer normalization is applied between the layers of each [Transformer Block](#4-transformer-block) and helps stabilize the distribution of activations by centering and scaling them to a standard normal distribution. This helps mitigate the problem of "Internal Covariate Shift" and facilitates the training of deeper networks.

Layer normalization is performed as follows:

- Computing the mean and standard deviation of activations across the feature dimension.
- Centering and scaling the activations based on the mean and standard deviation.
- Scaling and shifting the centered activations with learnable parameters to control normalization.
</details>

<details>
    <summary>4. Connections and Residual Connections</summary>

#### 4. Connections and Residual Connections
In addition to the submodules in a Transformer block, connections are added to retain information from previous layers. These connections can be implemented as residual connections, allowing activations to "flow through" the block unhindered and facilitating training.
Overall, the Transformer block enables the model to effectively process input data by capturing relationships between different parts of the input and learning richer representations of the data.
</details>
