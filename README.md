# firefly

A small PyTorch-style deep-learning library written in [Lugha](https://github.com/kithinjibrian/lugha), on top of [glowworm](https://github.com/kithinjibrian/glowworm)'s n-dimensional arrays.

```sh
lughac run example/xor.la        # a 2-layer network learns XOR
lughac run example/gradcheck.la  # every op's gradient vs. a finite difference
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
| `firefly.graph` | `param`, `input`, `value`, `grad`, `set_value`, `reset`; ops `add`, `sub`, `mul`, `matmul`, `add_row` (bias), `tanh`, `relu`, `sigmoid`, `sum`, `mean`; `backward`, `sgd` |
| `firefly.random` | `seeded(seed)` or `from_os()`, then `uniform()`, `normal(mean, std)`, `below(n)`, `shuffle(xs)`, `normals(shape, scale)`; a thin wrapper over Lugha's `std.random` (xoshiro256\*\*) |

## Roadmap

1. ✅ Tape autograd, SGD, XOR.
2. A bigram character model: embeddings, softmax cross-entropy, sampling text.
3. An MLP language model: context windows, Adam, train/validation split.
4. A tiny transformer: causal self-attention, layernorm, GELU — a mini GPT.

firefly depends on glowworm through `lugha.toml` (`glowworm = { path = "../glowworm", version = "0.1" }`), so clone both side by side.
