# The decoder-only transformer used by TinyLM-50

## Blocks and the residual stream

A decoder-only transformer is a stack of identical blocks operating on a
residual stream of shape (batch, sequence, hidden). Each block applies two
sub-layers, each preceded by normalisation and followed by a residual
addition. Pre-norm placement, where the normalisation is applied to the input
of the sub-layer rather than its output, is easier to train at depth because
the identity path from input to output is never rescaled.

TinyLM-50 uses RMSNorm. RMSNorm divides by the root mean square of the
activations and multiplies by a learned per-feature weight. It omits mean
subtraction, which saves a little work and performs comparably to LayerNorm in
this setting. Computation is done in float32 internally for stability, then
cast back to the input dtype.

## Grouped-query attention

Self-attention projects each token into queries, keys and values. In multi-head
attention each head has its own query, key and value projections. Grouped-query
attention (GQA) keeps the number of query heads but uses fewer key/value heads,
repeating each key/value head across a group of query heads. With 8 query heads
and 2 key/value heads, the KV cache is a quarter of the multi-head size while
query capacity is unchanged. This matters at inference: the cache grows with
sequence length and batch size.

Scaled dot-product attention computes softmax(QK^T / sqrt(d))V. A causal mask
prevents a position from attending to later positions; without it the model
could copy the token it is supposed to predict. PyTorch's
`scaled_dot_product_attention` accepts `is_causal=True` and chooses a fused
implementation when no explicit mask is supplied.

## Rotary position embeddings

RoPE encodes position by rotating pairs of query and key dimensions by an angle
proportional to the position. Because rotation is applied to both q and k, the
dot product depends on the relative offset between positions. The frequencies
are precomputed once; no learned position table is stored, so checkpoints are
smaller and the context length is a runtime choice within the precomputed
range. Applying RoPE to keys stored in the KV cache must use the correct
position offset, otherwise cached decoding produces different logits from a
single full forward pass.

## SwiGLU feed-forward network

Each block contains a position-wise feed-forward network. SwiGLU computes
`down(silu(gate(x)) * up(x))`. The gate and up projections both expand the
hidden dimension; the element-wise product acts as a learned gate. It has three
matrices instead of two, so the intermediate size is usually chosen smaller
than the classic 4x expansion. TinyLM-50 uses 1472 for a hidden size of 512,
which is about 2.9x and keeps the parameter budget near 49 million.

## Tied embeddings

The input token embedding is a matrix of shape (vocabulary, hidden). The output
head projects hidden states back to vocabulary logits. Tying the two matrices
saves vocabulary times hidden parameters, which is 8.2 million in a 16k by 512
model, and often improves small-model quality because both directions share
statistics. The trade-off is that the head cannot be quantised separately
without duplicating the table.
