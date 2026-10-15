# firefly

A small PyTorch-style deep-learning library written in [Lugha](https://github.com/kithinjibrian/lugha), on top of [glowworm](https://github.com/kithinjibrian/glowworm)'s n-dimensional arrays.

```sh
lughac run example/xor.la        # a 2-layer network learns XOR
lughac run example/gradcheck.la  # every op's gradient vs. a finite difference
lughac run example/bigram.la     # a character bigram model of names, then samples
lughac build -O2 example/mlp.la -o mlp && ./mlp   # an MLP language model (build with -O2: ~4× faster)
lughac build -O2 example/gpt.la -o gpt && ./gpt   # a tiny GPT: causal self-attention, layernorm, GELU
./gpt data/names-32k.txt 6000 32 4 0.1            # … on 32k names: data, steps, width, heads, decay
```

## How it works

Lugha has no shared references, so instead of tensors that point at each other, one `Graph` value is a **tape**: it holds every tensor's value, gradient and the op that made it, and a tensor is just its index. Methods take `mut self`, so the graph changes in place.

```
import glowworm.array;
import firefly.graph;
import firefly.random;

let mut rng = random.seeded(7);
let mut g = graph.new();
let w = g.param(rng.normals([2, 8], 1.0));   // parameters first
let b = g.param(array.zeros([8]));

for step in 0..1000 {
    g.reset();                                 // keep parameters, drop the rest
    let x = g.input(xs);
    let z = g.matmul(x, w);
    let a = g.add_row(z, b);
    let h = g.tanh(a);                         // … the rest of the model
    let loss = g.mean(h);
    g.backward(loss);
    g.sgd(0.1);
}
```

| Module | What's in it |
| --- | --- |
| `firefly.graph` | `param`, `input`, `value`, `grad`, `set_value`, `reset`; ops `add`, `sub`, `mul`, `matmul`, `add_row` (bias), `tanh`, `relu`, `sigmoid`, `sum`, `mean`, `rows` (embedding lookup), `cross_entropy` (stable softmax + negative log-likelihood), `reshape`; for transformers `transpose`, `scale`, `mul_row` (gain), `causal_softmax`, `layernorm`, `gelu`, `cols`/`concat` (attention heads); `backward`; optimisers `sgd`, `adam`, `adamw` (weight decay) |
| `firefly.text` | `from_lines(lines)` → a character `Vocab` (`.` = word boundary, id 0); `size`, `encode`, `decode` |
| `firefly.nn` | `softmax(xs)` |
| `firefly.random` | `seeded(seed)` or `from_os()`, then `uniform()`, `normal(mean, std)`, `below(n)`, `shuffle(xs)`, `pick(probs)` (weighted), `normals(shape, scale)`; a thin wrapper over Lugha's `std.random` (xoshiro256\*\*) |

## Roadmap

1. ✅ Tape autograd, SGD, XOR.
2. ✅ A bigram character model (`example/bigram.la`, on `data/names.txt`): reaches the counting optimum's loss, samples names.
3. ✅ An MLP language model (`example/mlp.la`): 3 characters of context, embeddings, a tanh layer, AdamW, minibatches, a train/validation split. On held-out names it beats the bigram model (validation loss 2.26 vs 2.41) and samples new name-like words.
4. ✅ A tiny transformer (`example/gpt.la`): token and position embeddings, one block of pre-layernorm causal multi-head self-attention and a GELU MLP with residual connections, AdamW. On held-out names it beats the bigram model (2.33 vs 2.51) and samples new names. Kept small (16 wide, 2 heads) for the tiny dataset.

Next: more data and more blocks, which mostly needs Lugha to stop copying arrays that ops only read (see Lugha's TODO).

## Data and results

- `data/names.txt`: 229 common names, for quick runs. Models overfit it quickly; weight decay keeps them honest.
- `data/names-32k.txt`: makemore's `names.txt` (32,032 names, from [karpathy/makemore](https://github.com/karpathy/makemore), MIT licence). `./mlp` and `./gpt` take a data file as their first argument.

On the 32k names, validation loss (lower is better):

| Model | Validation loss | Time (-O2) |
| --- | --- | --- |
| bigram (counted) | 2.455 | — |
| MLP, 3-char context (`./mlp data/names-32k.txt`, small-data settings) | 2.401 | 4 s |
| GPT, 32 wide, 4 heads, 1 block, 6000 steps (`./gpt data/names-32k.txt 6000 32 4 0.1`) | **2.242** | 1 m 30 s |

The GPT's training and validation losses match (2.241 vs 2.242): with 32k names it no longer memorises, so more width, more blocks and more steps would keep improving it. Samples: `kamori`, `mayzis`, `karie`, `adya`, `vayannata`.

Speed: since Lugha's milestone 55 (copy-on-write arrays), reading an array out of the graph (`let g = self.grads[i];`) no longer copies it, which made training 13–35% faster with identical results. Build examples with `-O2`.

firefly depends on glowworm through `lugha.toml` (`glowworm = { path = "../glowworm", version = "0.1" }`), so clone both side by side.
