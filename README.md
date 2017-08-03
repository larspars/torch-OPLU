# OPLU

A vectorized, CUDA compatible implementation of Orthogonal Permutation Linear Unit (OPLU) for Torch, as described in the paper "Norm-preserving Orthogonal Permutation Linear Unit Activation Functions (OPLU)", Chernodub et al, 2016

With the OPLU, every unit belongs to a pair `{x1, x2}`, and the activation function simply sorts this pair:
```lua
    f(x1, x2) = { max(x1, x2), min(x1, x2) } 
```
This activation function is norm and mean preserving, and has no diminishing effect on the gradient.
