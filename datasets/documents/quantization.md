# Weight-only quantisation for small language models

## Why quantise

A float32 parameter costs four bytes. A 49 million parameter model therefore
needs about 196 MB for its weights alone. Quantising weights to eight bits
reduces that to roughly 49 MB plus a small scale vector per output channel;
four bits reduce it to about 25 MB. For a model that must run offline on a
laptop, checkpoint size and memory bandwidth are often the binding constraints,
not raw arithmetic throughput.

## Symmetric per-channel quantisation

The simplest scheme is symmetric, per-output-channel, weight-only
quantisation. For a weight matrix W with rows corresponding to output features,
compute one scale per row:

    scale[r] = max(abs(W[r, :])) / qmax

Then store `q[r, c] = round(W[r, c] / scale[r])`, clamped to the signed range
of the target integer type. Dequantisation multiplies the integer back by the
scale. Per-channel scales cost one float per output feature, which is
negligible compared with the matrix, and they avoid the accuracy loss of a
single global scale when rows have very different magnitudes.

For INT8 the range is [-127, 127]. For INT4 the range is [-8, 7]; two signed
four-bit values are packed into one byte. Packing halves storage but adds
unpacking work: the inference path must extract the low and high nibbles and
sign-extend them before multiplying by the scale.

## What quantisation does and does not buy

Weight-only quantisation reduces checkpoint size and the memory needed to hold
weights. It does not automatically make CPU inference faster. A naive
implementation dequantises the weight matrix on every forward pass, which adds
computation and can be slower than float32 matrix multiplication. Real speedups
require fused integer kernels, which is why the honest claim for a simple
implementation is "smaller checkpoint, measured accuracy delta" rather than
"faster".

Activations stay in floating point in this scheme. Quantising activations
(dynamic quantisation) is a separate technique: it can speed up matrix
multiplication on CPUs that support integer instructions, but it requires
calibration and a runtime that supports the fused operation.

## Measuring the damage

Quantisation is an approximation, so it must be measured. The standard check is
validation cross-entropy or perplexity before and after quantisation on the
same batches, reported alongside the compression ratio. A small increase in
loss may be acceptable for a size-constrained deployment. A large increase, or
NaN outputs, means the scale calculation or the clamping range is wrong.

Embeddings and the language-model head are usually left in higher precision.
When the head is tied to the token embedding, quantising one would either break
the tie or duplicate the table, so leaving both in float is the simpler and
safer choice.
