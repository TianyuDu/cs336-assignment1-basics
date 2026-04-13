## 2,1 The Unicode Standard
### (a)
`chr(0)` gives `'\x00'` in Python.
$$
\boxed{\texttt{'\\x00'}}
$$
### (b)
The output of `__repr__` for a string is different from the output of `print`. When you use `print("x")`, it displays just the character `x`. In contrast, calling `"x".__repr__()` returns the string representation, which includes quotes: `'x'`.
$$
\boxed{\text{repr shows quotes and escapes, while print shows the rendered characters.}}
$$
### (c)
When we use `chr(0)` directly in Python, which is implicitly calling the `__repr__` method of the `str` class, it shows as `'\x00'`. However, when we use `print("some text", chr(0), "some text)`, the print function will interpret it as an empty string.
$$
\boxed{\text{The null byte is shown as '\\x00' in the representation, but prints as an invisible character.}}
$$

## 2.2 Uncode Encodings
### (a)
UTF-8 is more memory-efficient for text dominated by English characters and uses a smaller core set of byte values (256) compared to the much larger set required by encodings like UTF-16 (over 65,000 entries).
$$
\boxed{\text{UTF-8 is more space-efficient for mostly English text because ASCII characters take only 1 byte.}}
$$

### (b)
```pytyon
# Consider the following example:
>>> decode_utf8_bytes_to_str_wrong("é".encode("utf-8"))
# Raises UnicodeDecodeError because "é" encodes as b'\xc3\xa9' (two bytes),
# and neither 0xc3 nor 0xa9 is valid UTF-8 on its own.
```
$$
\boxed{\text{The wrong decoder fails because UTF-8 must decode the two-byte sequence together, not byte-by-byte.}}
$$

### (c)
`b'\xc3\x28'`
$$
\boxed{\texttt{b'\\xc3\\x28'}}
$$

## 2.5 Experimenting with BPE Tokenizer Training
Use `uv run python cs336_basics/25_train_bpe_tinystories.py` to execute launch the experiment.

Running on an Apple Silicon M1 Max with 64GB of RAM. Detailed profiling results:

```
Training BPE on /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt ...

========================================================================
TinyStories BPE Experiment
========================================================================
Data path: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt
Output directory: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_tinystories
Target vocab size: 10000
Special token: <|endoftext|>

Artifacts
- vocab.json: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_tinystories/vocab.json
- merges.txt: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_tinystories/merges.txt
- report.md: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_tinystories/report.md
- report.json: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_tinystories/report.json

Training summary
- Input size: 2227753162 bytes (2.07 GiB)
- Total wall-clock time: 64.2s
- Approx peak memory (sampled process-tree RSS): 8.87 GiB (9302048 KiB)
- Requested workers: 5
- Worker processes used: 5
- Number of chunks: 5
- Final vocab size: 10000
- Merges learned: 9743
- Special token present in vocab: True
- Unique pretokens: 59933
- Unique token sequences: 59933
- Initial distinct adjacent pairs: 2108

Longest token
- Bytes repr: b' accomplishment'
- Decoded text: ' accomplishment'
- Length: 15 bytes

Stage timings
- Read input: 0.3s (0.5% of total)
- Find chunk boundaries: 0.0s (0.0% of total)
- Pre-tokenize documents: 56.2s (87.5% of total)
- Build token sequences: 0.1s (0.1% of total)
- Build pair counts + heap: 0.1s (0.2% of total)
- Run BPE merge loop: 7.5s (11.7% of total)
- Bottleneck: Pre-tokenize documents at 56.2s (87.5% of total)
```

### `train_bpe_tinystories` (a)
The run took 64.2 seconds and used about 8.87 GiB at peak; the longest learned token was `b' accomplishment'` (15 bytes), which seems reasonable.

### `train_bpe_tinystories` (b)
The slowest part by far is pretokenization, which took 56.2 seconds. In comparison, the actual BPE merge step was done in just 7.5 seconds.


### `train_bpe_expts_owt` (a) and (b)

The raw training log shows the following.
```
========================================================================
OpenWebText BPE Experiment
========================================================================
Data path: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/owt_train.txt
Output directory: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_owt
Target vocab size: 32000
Special token: <|endoftext|>

Artifacts
- vocab.json: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_owt/vocab.json
- merges.txt: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_owt/merges.txt
- report.md: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_owt/report.md
- report.json: /Users/tianyudu/Development/CS336/cs336-assignment1-basics/data/bpe_owt/report.json

Training summary
- Input size: 11920511059 bytes (11.10 GiB)
- Total wall-clock time: 36711.2s
- Approx peak memory (sampled process-tree RSS): 21.38 GiB (22415984 KiB)
- Requested workers: 5
- Worker processes used: 5
- Number of chunks: 5
- Final vocab size: 32000
- Merges learned: 31743
- Special token present in vocab: True
- Unique pretokens: 6601892
- Unique token sequences: 6601892
- Initial distinct adjacent pairs: 19592

Longest tokens
- 1. b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' decoded as '\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2' (64 bytes)
- 2. b'----------------------------------------------------------------' decoded as '----------------------------------------------------------------' (64 bytes)
- 3. b'\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94' decoded as '\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014' (48 bytes)
- 4. b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' decoded as '\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2\\xc3\\xc2' (32 bytes)
- 5. b'________________________________' decoded as '________________________________' (32 bytes)
- 6. b'================================' decoded as '================================' (32 bytes)
- 7. b'................................' decoded as '................................' (32 bytes)
- 8. b'--------------------------------' decoded as '--------------------------------' (32 bytes)
- 9. b'********************************' decoded as '********************************' (32 bytes)
- 10. b'\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94' decoded as '\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014\\u2014' (24 bytes)

Stage timings
- Read input: 4.5s (0.0% of total)
- Find chunk boundaries: 0.0s (0.0% of total)
- Pre-tokenize documents: 306.0s (0.8% of total)
- Build token sequences: 10.9s (0.0% of total)
- Build pair counts + heap: 67.4s (0.2% of total)
- Run BPE merge loop: 36322.4s (98.9% of total)
- Bottleneck: Run BPE merge loop at 36322.4s (98.9% of total)
```


**(a)** The longest OpenWebText tokens are 64 bytes long: one is a repeated mojibake byte pattern (`\xc3\xc2` repeated), and another is a run of 64 hyphens. This makes sense for byte-level BPE on noisy web text, which can merge frequent formatting sequences and encoding artifacts as well as ordinary words. The overall memory cost ranges from 23GiB to 34GiB over the approximately 10 hours of training.

**(b)** Compared with TinyStories, the OpenWebText tokenizer learns a much noisier vocabulary, including long punctuation runs and encoding-artifact tokens, whereas TinyStories learns cleaner, more word-like tokens such as ` accomplishment`. This suggests TinyStories is more regular and linguistically simple, while OpenWebText is more heterogeneous and contains more formatting noise.

## 2.7 Experiments
### `tokenizer_expeirments` (a) (b) and (c)
Detailed log:
```
(cs336-basics) ➜  cs336-assignment1-basics git:(main) ✗ uv run cs336_basics/27_tokenizer_experiments.py
sample_size=10 seed=0
TinyStories docs: 332, 2122, 2485, 2934, 3156, 3318, 3446, 3905, 3981, 4189 (sampled from 5061 documents)
OpenWebText docs: 65, 121, 138, 262, 508, 583, 780, 783, 822, 868 (sampled from 879 documents)
TinyStories tokenizer (10K) compression_ratio_bytes/token: tinystories=4.1357 openwebtext=3.4112 combined=3.4986 sample_throughput_bytes_per_s=914182B/s
OpenWebText tokenizer (32K) compression_ratio_bytes/token: tinystories=4.0625 openwebtext=4.5460 combined=4.4702 sample_throughput_bytes_per_s=864358B/s
part_b_tinystories_on_owt_bytes_per_token=3.411225
part_b_tinystories_on_tinystories_bytes_per_token=4.135738
part_b_delta_bytes_per_token=-0.724513
part_c_ts_bytes_per_second=914182
part_c_ts_hours_for_825GB=269.17
part_c_owt_bytes_per_second=864358
part_c_owt_hours_for_825GB=284.68
```

**(a)** The TinyStories tokenizer (10K) compression ratio is 4.1357 bytes/token on TinyStories and 3.4112 bytes/token on OpenWebText. The OpenWebText tokenizer (32K) compression ratio is 4.0625 bytes/token on TinyStories and 4.5460 bytes/token on OpenWebText.

**(b)** The TinyStories tokenizer on OpenWebText achieves a compression ratio of 3.4112 bytes/token, which is lower than the TinyStories-domain ratio of 4.1357 bytes/token. The delta is -0.724513 bytes/token.

**(c)** The estimated tokenizer throughput for the TinyStories corpus is 914182 bytes/second, which will take approximately 269.17 hours to tokenize 825GB of data. The estimated tokenizer throughput for the OpenWebText corpus is 864358 bytes/second, which will take approximately 284.68 hours to tokenize 825GB of data.

**(d)**
uint16 is an appropriate choice for storing token IDs because the vocabulary size (32K) fits well within the 16-bit range (0 to 65,536), without overflow or wasting memory.

## 3.5 The Full Transformer LM `transformer_accounting`
### (a)
- Vocab size: $V = 50{,}257$,
- Context length: $C = 1{,}024$,
- Number of layers: $N = 48$,
- Model dimension: $d_{model} = 1600$,
- Number of heads: $K = 25$,
- Feed-forward dimension: $d_{ff} = 4288$.

- **Token embedding layer**: $V \times d_{model} = 50{,}257 \times 1600 = 80{,}411{,}200$ parameters.
- **Each transformer block**:
    - Two normalization layers: $2 \times d_{model} = 2 \times 1600 = 3{,}200$ parameters.
    - **Causal multi-head attention layer**: 4 linear layers for the Q, K, V, and output projections, each with parameter count $d_{model} \times d_{model}$ and no bias: total parameters: $4 \times (d_{model} \times d_{model}) = 4 \times (1600 \times 1600) = 10{,}240{,}000$.
    - RoPE positional encoding: determinstic, no parameters.
    - Position-wise Feedforward (SwiGLU): 3 weight matrices—
        - W1: shape $[d_{ff},\, d_{model}] = [4288,\, 1600]$
        - W2: shape $[d_{model},\, d_{ff}] = [1600,\, 4288]$
        - W3: shape $[d_{ff},\, d_{model}] = [4288,\, 1600]$
      Each is a matrix of parameters with no bias. Total parameter count for this sublayer:
      $3 \times (d_{ff} \times d_{model}) = 3 \times (4288 \times 1600) = 20{,}582{,}400$.
    - Total parameters per block: $3{,}200 + 10{,}240{,}000 + 20{,}582{,}400 = 30{,}825{,}600$.
- **All 48 transformer blocks**: $48 \times 30{,}825{,}600 = 1{,}479{,}628{,}800$ parameters.
- **Final normalization layer**: $d_{model} = 1600$ parameters.
- **Language model head**: $d_{model} \times V = 1600 \times 50{,}257 = 80{,}411{,}200$ parameters.
- **Softmax**: no parameters.

**Total number of parameters**: $80{,}411{,}200 + 1{,}479{,}628{,}800 + 1{,}600 + 80{,}411{,}200 = 1{,}640{,}452{,}800$.

Assuming each parameter is represented using single-precision floating point (float32), each parameter is 4 bytes, so the total memory usage is $1{,}640{,}452{,}800 \times 4 = 6{,}561{,}811{,}200$ bytes, which is approximately 6.56 GB.
$$
\boxed{1{,}640{,}452{,}800\text{ parameters},\quad 6.56\text{ GB in float32}.}
$$

### (b)
With input size $C = 1024$, assume inference time so batch size $B = 1$. Since the prompt asks for **matrix multiplies**, I count the matrix multiplications in the forward pass and do not include embedding lookups, RMSNorm, softmax, masking, RoPE elementwise rotations, or residual additions in the main FLOP total.

- Embedding lookup layer: just a lookup, so $0$ matrix-multiply FLOPs.
- For **each** transformer block:
    - **Causal multi-head attention**:
        - QKV projection: in the code, the Q/K/V weights are concatenated and applied as one `F.linear`, so this is a multiply of shape $[C, d_{model}] \times [d_{model}, 3d_{model}]$, which costs
          $2 \cdot C \cdot d_{model} \cdot (3d_{model}) = 6 C d_{model}^2$ FLOPs.
        - Attention scores $QK^\top$: across all heads, this costs $2 C^2 d_{model}$ FLOPs.
        - Weighted sum $\mathrm{probs} \cdot V$: across all heads, this costs another $2 C^2 d_{model}$ FLOPs.
        - Output projection: multiply of shape $[C, d_{model}] \times [d_{model}, d_{model}]$, which costs $2 C d_{model}^2$ FLOPs.
        - Total attention FLOPs per block:
          $6 C d_{model}^2 + 2 C^2 d_{model} + 2 C^2 d_{model} + 2 C d_{model}^2 = 8 C d_{model}^2 + 4 C^2 d_{model}$.
    - **RoPE**: no matrix multiplies. In this codebase, RoPE only does elementwise multiply/add operations on $Q$ and $K$ using cached sine/cosine tables, so it contributes $0$ matrix-multiply FLOPs.
    - **Position-wise Feedforward (SwiGLU)**:
        - $xW_1$: shape $[C, d_{model}] \times [d_{model}, d_{ff}]$, so $2 C d_{model} d_{ff}$ FLOPs.
        - $xW_3$: shape $[C, d_{model}] \times [d_{model}, d_{ff}]$, so another $2 C d_{model} d_{ff}$ FLOPs.
        - $\mathrm{slu\_inner} W_2$: shape $[C, d_{ff}] \times [d_{ff}, d_{model}]$, so $2 C d_{ff} d_{model}$ FLOPs.
        - Total FFN FLOPs per block:
          $6 C d_{model} d_{ff}$.

- Final normalization: no matrix multiplies.
- Language model prediction head: logits are computed by multiplying $[C, d_{model}] \times [d_{model}, V]$, so this costs
  $2 C d_{model} V$ FLOPs.

Therefore, the total matrix-multiply FLOPs for one forward pass are
$N \left(8 C d_{model}^2 + 4 C^2 d_{model} + 6 C d_{model} d_{ff}\right) + 2 C d_{model} V$ FLOPs, where $N$ is the number of transformer blocks. For GPT-2 XL, this is approximately $3.52 \times 10^{12}$ FLOPs.
$$
\boxed{3.52 \times 10^{12}\text{ forward FLOPs for GPT-2 XL at }C=1024.}
$$

### (c)
The largest FLOP contributor is the position-wise feedforward (SwiGLU) sublayer, since it contributes $6 C d_{model} d_{ff}$ FLOPs per block and $d_{ff}$ grows roughly linearly with $d_{model}$. The attention projection matrices are the next-largest contributor, while the $QK^\top$ and $PV$ attention matmuls take a smaller share at this context length.
$$
\boxed{\text{The SwiGLU feedforward sublayer is the largest FLOP contributor.}}
$$

### (d)
Using the same matrix-multiply accounting as in part (b), with $C = 1024$ and $V = 50{,}257$, the forward-pass FLOP breakdown is:

| Model | $d_{ff}$ | Attention projections $(QKV + O)$ | Attention score/value matmuls $(QK^\top + PV)$ | FFN (SwiGLU) | LM head | Total |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 small | 2048 | $5.80 \times 10^{10}$ (19.88%) | $3.87 \times 10^{10}$ (13.25%) | $1.16 \times 10^{11}$ (39.76%) | $7.90 \times 10^{10}$ (27.10%) | $2.92 \times 10^{11}$ |
| GPT-2 medium | 2752 | $2.06 \times 10^{11}$ (24.83%) | $1.03 \times 10^{11}$ (12.42%) | $4.16 \times 10^{11}$ (50.05%) | $1.05 \times 10^{11}$ (12.70%) | $8.30 \times 10^{11}$ |
| GPT-2 large | 3392 | $4.83 \times 10^{11}$ (27.32%) | $1.93 \times 10^{11}$ (10.93%) | $9.60 \times 10^{11}$ (54.30%) | $1.32 \times 10^{11}$ (7.45%) | $1.77 \times 10^{12}$ |
| GPT-2 XL | 4288 | $1.01 \times 10^{12}$ (28.62%) | $3.22 \times 10^{11}$ (9.16%) | $2.02 \times 10^{12}$ (57.53%) | $1.65 \times 10^{11}$ (4.68%) | $3.52 \times 10^{12}$ |

As model size increases at fixed context length, a greater portion of FLOPs go to the SwiGLU feedforward and attention projection layers, while the attention score/value matmuls and lm_head contribute less.
$$
\boxed{\text{As model size grows, FLOPs shift toward the FFN and attention projections.}}
$$

### (e)
At $C = 16{,}384$, the forward-pass FLOP breakdown for GPT-2 XL is:

| Model | $d_{ff}$ | Attention projections $(QKV + O)$ | Attention score/value matmuls $(QK^\top + PV)$ | FFN (SwiGLU) | LM head | Total |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 XL ($C = 16{,}384$) | 4288 | $1.61 \times 10^{13}$ (12.06%) | $8.25 \times 10^{13}$ (61.73%) | $3.24 \times 10^{13}$ (24.24%) | $2.63 \times 10^{12}$ (1.97%) | $1.34 \times 10^{14}$ |

This is about a $38\times$ increase over the $C = 1024$ case, and the attention score/value matmuls now dominate because they scale quadratically with context length.
$$
\boxed{1.34 \times 10^{14}\text{ forward FLOPs at }C=16{,}384,\text{ dominated by }QK^\top\text{ and }PV.}
$$

## 4.2 The SGD Optimizer
### `learning_rate_tuning`
The results are:
```
(dev) ➜  cs336-assignment1-basics git:(main) ✗ uv run cs336_basics/learning_rate_tuning.py
lr=1: [26.271406, 25.231056, 24.522461, 23.959410, 23.482616, 23.064425, 22.689322, 22.347588, 22.032663, 21.739874]
lr=1e+01: [26.271406, 16.813700, 12.394342, 9.697249, 7.854772, 6.512506, 5.492435, 4.693442, 4.053156, 3.530750]
lr=1e+02: [26.271406, 26.271404, 4.507460, 0.107874, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
lr=1e+03: [26.271406, 9483.976562, 1638032.000000, 182213552.000000, 14759298048.000000, 931480731648.000000, 47819176017920.000000, 2057385568894976.000000, 75830843066548224.000000, 2435012535733190656.000000]
```
With 10 steps, `lr=1` reduces the loss slowly, `lr=1e1` is much faster, and `lr=1e2` quickly reaches zero after a brief overshoot. `lr=1e3` is too large and causes immediate divergence.
$$
\boxed{\text{lr}=1\text{ decays slowly, }10\text{ decays faster, }100\text{ overshoots then converges, and }1000\text{ diverges.}}
$$

## 4.3 AdamW
Let $B = \text{batch\_size}$, $V = \text{vocab\_size}$, $C = \text{context\_length}$, $N = \text{num\_layers}$, $d = d_{\text{model}}$, and $H = \text{num\_heads}$. Per the prompt, assume $d_{ff} = \frac{8}{3}d$.

### (a)
The total number of model parameters is
$$
P = 2Vd + N(4d^2 + 3dd_{ff} + 2d) + d.
$$
Using $d_{ff} = \frac{8}{3}d$, this simplifies to
$$
P = 2Vd + N(12d^2 + 2d) + d.
$$

So the memory used by parameters, gradients, and AdamW optimizer state is:
- Parameters: $4P$ bytes
- Gradients: $4P$ bytes
- Optimizer state: $8P$ bytes, since AdamW stores first-moment and second-moment tensors (`m` and `v`)

For activations, I count the outputs of the components listed in the prompt.

For one transformer block:
- RMSNorms: $2BCd$
- QKV projections: $3BCd$
- $QK^\top$ matrix multiply and softmax: $2BHC^2$
- Weighted sum of values and output projection: $2BCd$
- SwiGLU: $W_1$, $W_3$, SiLU gate output, and element-wise product contribute $4BCd_{ff}$
- SwiGLU $W_2$: $BCd$

So one block stores
$$
B(8Cd + 4Cd_{ff} + 2HC^2)
$$
activations, which becomes
$$
B\left(\frac{56}{3}Cd + 2HC^2\right)
$$
after substituting $d_{ff} = \frac{8}{3}d$.

Adding the final RMSNorm, output embedding, and cross-entropy on the logits gives total activation memory
$$
A = B\left[N\left(8Cd + 4Cd_{ff} + 2HC^2\right) + Cd + 2CV + C\right]
$$
elements, or equivalently
$$
A = B\left[N\left(\frac{56}{3}Cd + 2HC^2\right) + Cd + 2CV + C\right].
$$
The extra $C$ term is the per-token loss tensor after cross-entropy.

Since each activation is float32, activation memory is $4A$ bytes. Therefore the peak AdamW training memory is
$$
M_{\text{peak}} = 4P + 4A + 4P + 8P = 16P + 4A \text{ bytes}.
$$
$$
\boxed{M_{\text{peak}} = 16P + 4A \text{ bytes}.}
$$

### (b)
For a GPT-2 XL-shaped model, use
$$
V = 50{,}257,\quad C = 1024,\quad N = 48,\quad d = 1600,\quad H = 25.
$$
Then
$$
P = 2Vd + N(12d^2 + 2d) + d = 1{,}635{,}537{,}600.
$$

So:
- Parameters: $4P = 6{,}542{,}150{,}400$ bytes
- Gradients: $4P = 6{,}542{,}150{,}400$ bytes
- Optimizer state: $8P = 13{,}084{,}300{,}800$ bytes

The activation term becomes
$$
4A = 16{,}356{,}618{,}240 \cdot B \text{ bytes}.
$$

Therefore the peak memory is
$$
M_{\text{peak}}(B) = 16{,}356{,}618{,}240 \cdot B + 26{,}168{,}601{,}600 \text{ bytes}.
$$
In decimal GB, this is approximately
$$
M_{\text{peak}}(B) \approx 16.36B + 26.17 \text{ GB}.
$$

Setting this less than or equal to $80$ GB gives
$$
B \le \frac{80 - 26.17}{16.36} \approx 3.29,
$$
so the maximum batch size that still fits is
$$
\boxed{M_{\text{peak}}(B) \approx 16.36B + 26.17\text{ GB},\quad B_{\max}=3.}
$$

### (c)
For one AdamW step, each parameter participates in:
- decoupled weight decay: about $2$ FLOPs per parameter
- first-moment update: about $3$ FLOPs per parameter
- second-moment update: about $4$ FLOPs per parameter
- final normalized parameter update: about $5$ FLOPs per parameter

Ignoring the $O(1)$ scalar work for bias correction, this is about
$$
14P
$$
FLOPs per optimizer step, i.e.
$$
14\left[2Vd + N(12d^2 + 2d) + d\right].
$$

For GPT-2 XL-shaped dimensions, this is
$$
14P = 22{,}897{,}526{,}400 \approx 2.29 \times 10^{10}
$$
FLOPs.
$$
\boxed{14P \approx 2.29 \times 10^{10}\text{ FLOPs per AdamW step.}}
$$

### (d)
Using the same matrix-multiply accounting as in Section 3.5 and substituting $d_{ff} = \frac{8}{3}d$, the forward FLOPs for one sequence are
$$
F_{\text{fwd}} = N(24Cd^2 + 4C^2d) + 2CdV.
$$
For GPT-2 XL-shaped values, this gives
$$
F_{\text{fwd}} = 3.5067 \times 10^{12} \text{ FLOPs}.
$$

Assuming the backward pass costs twice the forward pass, the total training FLOPs per sequence are
$$
F_{\text{train}} = 3F_{\text{fwd}} = 1.0520 \times 10^{13}.
$$
Dividing by context length gives FLOPs per token:
$$
\frac{F_{\text{train}}}{C} = 1.02735 \times 10^{10} \text{ FLOPs/token}.
$$

At 50% MFU on a single H100, the effective throughput is
$$
0.5 \times 495 \times 10^{12} = 2.475 \times 10^{14} \text{ FLOPs/s}.
$$
So the token throughput is
$$
\frac{2.475 \times 10^{14}}{1.02735 \times 10^{10}} \approx 2.41 \times 10^4 \text{ tokens/s}.
$$

Using batch size $1024$ and context length $1024$, the total number of training tokens over $400{,}000$ steps is
$$
400{,}000 \times 1024 \times 1024 = 4.1943 \times 10^{11} \text{ tokens}.
$$
Therefore the total training time is
$$
\frac{4.1943 \times 10^{11}}{2.4091 \times 10^4}
\approx 1.741 \times 10^7 \text{ s}
\approx 4.84 \times 10^3 \text{ hours}
\approx 201.5 \text{ days}.
$$

So the training time is approximately
$$
\boxed{4.84 \times 10^3 \text{ hours} \approx 201.5 \text{ days}.}
$$


# 7. Experiments
## 7.1 How to Run Experiments and Deliverables
### `experiment_log`
I set up W&B logging directly in `training_together.py` so that every run automatically saves important experiment details—like the model architecture, optimization settings, dataset and checkpoint used, computing environment, and git state—but skips uploading any big model files. Throughout training and validation, the script regularly logs loss curves (both for gradient steps and actual wall-clock time), plus stats such as learning rate, training speed, gradient norms, timings, memory usage, and checkpoint info.,

### `learning_rate` (a)
To visualize the effect of different learning rates on training, I used `scripts/723_learning_rate_tuning.py` to sweep nine values. The figure below shows the same overall trend on the displayed runs (`1e-5`, `3e-5`, `1e-4`, `3e-4`, and `1e-3`): larger learning rates in this range converged faster and ended with lower validation loss, with final values of about `2.44`, `1.80`, `1.65`, `1.49`, and `1.43`, respectively. The model with `1e-3` performed best, achieveing the lowest final validation loss of `1.43`. Each training session took around 23 minutes to complete.
![Learning rate sweep results](figures/7.2.3a_learning_rate.png)

### `learning_rate` (b) more exploration
When the learning rate becomes too large (e.g., 0.02), the training loss becomes instable and fail to converge to a reasonly low value.

![Learning rate sweep results](figures/7.2.3b_learning_rate_explode.png)

When I try a large learning rate (1.0), the validation loss explodes and the training stopped due to numerical overflow.
![Learning rate sweep results](figures/7.2.3b_explode_2.png)

For reference, I have also attached the training loss curve below.
![Learning rate sweep results](figures/7.2.3_training_loss.png)

During this set of experiments, when the learning rate is too small (e.g., 1e-5), the training loss was converging slowly but the validation loss failed to converge to a reasonable value before the computational budget was exhausted. At a balanced point, the learning rate of 1e-3 performed best, achieving the lowest final validation loss of 1.43, below the 1.45 requirement. When the learning rate was too large, the training loss became instable and numerical overflow was encountered.

### `batch_size_experiment`
I have experiment with batch size of 32, 64, 128, 256, 512, 1024. The figure below shows the validation loss curve for each batch size. Since the total number of tokens procssed is fixed, a larger batch size results in fewer training steps. At a batch size of 1024, the GPU memory usage was around 155GiB, indiciating that this is clos to the B200's memory limit, use a higher batch size of 2048 will lead to OOM.

With a batch size of 1024, the model was clearly under-fitting, the validation loss stopped at 1.52, with a small batch size of 32, the gradient update became too noisy and the validation loss was oscillated around 1.45, and landed at 1.48. With a balanced batch size between 64 and 256, the validation loss was optimal at 1.42.

![Batch size experiment results](figures/7.2.3_batch_size.png)


### `generate`
Please see the following command and output for generating text from the checkpoint. Note that the assignment did not explicitly speficy the decoding parameters or the prompt, so I was using the 'Once upon a time' prompt and tried a few parameters. The following one is the result I found reasonable.
```
(cs336-basics) (base) ➜  cs336-assignment1-basics git:(main) ✗ uv run python scripts/generate.py \
  --checkpoint-path checkpoints/723-learning-rate-tuning/0_001.pt \
  --vocab-path data/bpe_tinystories/vocab.json \
  --merges-path data/bpe_tinystories/merges.txt \
  --prompt "Once upon a time" \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.95
Loading tokenizer from data/bpe_tinystories/vocab.json ...
Building model on mps ...
Loading checkpoint from checkpoints/723-learning-rate-tuning/0_001.pt ...
Generating up to 256 tokens (temp=0.8, top_p=0.95) ...

--- Config ---
checkpoint : checkpoints/723-learning-rate-tuning/0_001.pt
iteration  : 40000
device     : mps
dtype      : float32
temperature: 0.8
top_p      : 0.95
seed       : 0
prompt     : 'Once upon a time'
new tokens : 242
stopped_eos: True

--- Generated text ---

Once upon a time, in a small town, there lived a boy named Tim. Tim had a toy gun that he loved to play with. One day, Tim saw a big box in his room. He was very happy and thought it would be fun to play with his gun.
Tim played with his gun outside. He pretended to shoot at trees and flowers. He had so much fun that he didn't want to stop. But then, Tim saw a big rock in his way. He knew he had to be careful not to step on it.
Tim started to play with his toy gun. He pretended to shoot at the big rock. He was having a lot of fun. But then, Tim got too close to the rock. The rock hit the big rock and broke into pieces. Tim was very sad and scared. He cried and ran inside.
His mom saw what happened and talked to Tim. She told him not to play with the gun that could break. Tim learned that he should listen to his mom and not play with things that are not safe. From that day on, Tim always played with his toy gun and safe toys, but he never went outside to play in the dark.
```

This run generated 242 new tokens before emitting the end-of-sequence token, so it satisfies the requirement of generating up to 256 tokens or stopping at the first `<|endoftext|>` token. Overall, the output is fairly fluent: most sentences are grammatical, easy to read, and consistent with the simple TinyStories style. However, the story is not very coherent over longer stretches. It repeats phrases such as "big rock" and "play with his gun," and some transitions are awkward or do not make much sense.

At least two factors affect the quality of this sample. First, model capacity and checkpoint quality matter: with a relatively small model and limited training, the model can learn local grammar and short-range patterns without fully learning long-range coherence. Second, the decoding parameters matter: using `temperature=0.8` and `top-p=0.95` helps produce more varied continuations, but it also makes the output more likely to drift, repeat itself, or introduce inconsistencies.
