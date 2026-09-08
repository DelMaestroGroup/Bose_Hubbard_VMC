# VMCBoseHubbard

Variational Monte Carlo for the Bose–Hubbard model: Gutzwiller and Jastrow
wavefunctions on any lattice given by its adjacency, optimized by stochastic
reconfiguration, with per-bond hopping amplitudes.

This package is the variational back end of
[Pigsfli.jl](https://github.com/DelMaestroGroup/Pigsfli.jl), which uses it to
optimize the guiding states its projector Monte Carlo starts from. It is
usable on its own; the API reference and the derivations are still being
written, and `documentation/documentation.pdf` is the current description of
the method.

## Installation

Julia 1.11 or later. From the repository root:

```julia
] activate VMCBoseHubbard
] instantiate
```

The package lives in the `VMCBoseHubbard/` subdirectory; a project that depends
on it adds it with that `subdir`, as Pigsfli's `Project.toml` does.

## Examples

`VMCBoseHubbard/examples/gutzwiller_benchmark.jl` and `jastrow_benchmark.jl`
optimize each wavefunction over a range of interaction strengths and compare
against exact results.
