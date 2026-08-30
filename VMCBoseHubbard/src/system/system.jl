# system.jl

#=
Purpose: store system parameters and other system information
Input: t (hopping amplitude), U (interaction strength), N (total number of particles), lattice (lattice struct)
Author: Will Mallah
Last Updated: 05/22/2026
=#

#=
BOND-RESOLVED HOPPING
─────────────────────
`t` is either a scalar, the uniform case, or a per-bond amplitude matrix t[i,j]
for a lattice whose bonds differ by direction (an anisotropic square, a ladder,
a brick-wall honeycomb, …). Both flow through `hop_amplitude` below, so the
local energies read the amplitude for the bond they are actually on and every
existing scalar call keeps its old behaviour exactly.

`t` and `U` carry SEPARATE type parameters. They shared one before, which is
why a matrix t was not expressible: it would have forced U to be a matrix too.
=#
struct System{Tt, Tu <: Real}
    t::Tt
    U::Tu
    N::Int
    lattice::AbstractLattice
end

"""
    hop_amplitude(t, i, j) -> Real

Hopping amplitude on the bond (i, j).

A scalar `t` is uniform and ignores the bond; a matrix is indexed by it. Keeping
this behind one inlined function means the local energies do not branch on the
representation, and a new representation (a dict, a callable) only has to add a
method here.
"""
@inline hop_amplitude(t::Real, ::Int, ::Int) = t
@inline hop_amplitude(t::AbstractMatrix, i::Int, j::Int) = @inbounds t[i, j]
@inline hop_amplitude(t, i::Int, j::Int) = t(i, j)   # callable, for anything else
