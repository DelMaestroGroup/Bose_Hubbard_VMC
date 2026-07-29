abstract type AbstractLattice end

# ─── General lattice ───────────────────────────────────────────────────────────
#=
Purpose: represent any lattice as (i) a hopping adjacency list and (ii) a
         translational-symmetry classification of every ordered site pair into
         "displacement shells", used to parametrize the anisotropic real-space
         Jastrow potential v(shell).

Fields
  neighbors  : adjacency list (hopping bonds); lattice-agnostic MC uses only this.
  pair_shell : L×L matrix; pair_shell[i,j] = the 1-based shell index of the
               ordered pair (i,j). pair_shell[i,j] == pair_shell[j,i] (the Jastrow
               is symmetric, so the displacement Δ and −Δ share a shell). Shells are
               ordered by |Δ|² then key, so the on-site (Δ=0) shell(s) come first —
               pair_shell[i,i] == 1 on a single-sublattice lattice (multi-sublattice
               lattices have one on-site shell per sublattice; always use
               pair_shell[i,i] for site i's on-site term, never a hard-coded 1).
  n_shells   : number of distinct shells = length of the Jastrow vector vr.
  shell_reps : per-shell canonical key (folded integer displacement + sublattice
               pair) — the geometric meaning of each vr component, for the vr file.

The classification uses INTEGER lattice coordinates (unit-cell offsets + a
sublattice index) with an integer minimal-image fold, so it is exact (no float
thresholds) and reproducible byte-for-byte on the Pigsfli side.
=#
struct GeneralLattice <: AbstractLattice
    neighbors  ::Vector{Vector{Int}}
    pair_shell ::Matrix{Int}
    n_shells   ::Int
    shell_reps ::Vector{NTuple{N,Int} where N}
end

# Minimal-image representative of an integer cell offset d on a periodic axis of
# length L, folded to (−L/2, L/2]. period == 0 means an open axis (no fold).
@inline function _min_image(d::Int, L::Int)
    L == 0 && return d
    r = mod(d, L)
    return r > L ÷ 2 ? r - L : r
end

# Canonical key for the ordered pair with cell displacement Δ (per axis) between
# sublattices (si → sj): the lexicographically smaller of (Δ, si, sj) and its
# reflection (−Δ, sj, si), so v(shell) = v(−shell) is enforced by construction.
function _shell_key(Δ::Vector{Int}, si::Int, sj::Int, periods::Vector{Int})
    D = length(Δ)
    m  = ntuple(a -> _min_image(Δ[a],  periods[a]), D)
    m2 = ntuple(a -> _min_image(-Δ[a], periods[a]), D)
    key1 = (m...,  si, sj)
    key2 = (m2..., sj, si)
    return min(key1, key2)
end

#=
Purpose: build a GeneralLattice from integer site coordinates.
Input
  neighbors : hopping adjacency list (length L).
  cells     : D×L integer unit-cell offsets (column i = coordinates of site i).
  basis     : length-L sublattice index per site (0 for single-site unit cells).
  periods   : length-D period per axis (0 = open/no wrap).
Output: GeneralLattice with pair_shell / shell_reps / n_shells filled in.
=#
function GeneralLattice(neighbors::Vector{Vector{Int}},
                        cells::Matrix{Int},
                        basis::Vector{Int},
                        periods::Vector{Int})
    L = length(neighbors)
    @assert size(cells, 2) == L "cells must have one column per site"
    @assert length(basis) == L

    # First pass: canonical key for every ordered pair.
    keys = Matrix{Any}(undef, L, L)
    Δ = Vector{Int}(undef, size(cells, 1))
    for i in 1:L, j in 1:L
        @views Δ .= cells[:, j] .- cells[:, i]
        keys[i, j] = _shell_key(Δ, basis[i], basis[j], periods)
    end

    # Assign shell indices deterministically, ordered by squared displacement
    # length then by the canonical key. This puts the on-site (Δ=0) shell(s) first
    # and, on a single-sublattice lattice, orders by distance — so a 1D chain
    # reproduces the old ring-distance ordering (r = 0,1,…,⌊L/2⌋) exactly.
    _dist2(k) = (D = length(k) - 2; sum(k[a]^2 for a in 1:D))
    uniq = sort(unique(vec(keys)); by = k -> (_dist2(k), k))
    index_of = Dict(k => s for (s, k) in enumerate(uniq))
    pair_shell = [index_of[keys[i, j]] for i in 1:L, j in 1:L]
    shell_reps = Vector{NTuple{N,Int} where N}(undef, length(uniq))
    for (s, k) in enumerate(uniq)
        shell_reps[s] = k
    end
    return GeneralLattice(neighbors, pair_shell, length(uniq), shell_reps)
end

# ─── Convenience constructors for the built-in test geometries ─────────────────
#=
1D chain. periodic=true wraps; the shells reduce to ring distances 0,1,…,⌊L/2⌋,
so a chain built this way is byte-for-byte backward compatible with the old
1D Jastrow (vr indexed by r = min(|i−j|, L−|i−j|)).
=#
function Lattice1D(L::Int; periodic::Bool=true)
    neighbors = [Int[] for _ in 1:L]
    for i in 1:(L-1)
        push!(neighbors[i], i+1); push!(neighbors[i+1], i)
    end
    if periodic && L > 2
        push!(neighbors[1], L); push!(neighbors[L], 1)
    end
    cells  = reshape(collect(0:L-1), 1, L)
    basis  = zeros(Int, L)
    period = [periodic ? L : 0]
    return GeneralLattice(neighbors, cells, basis, period)
end

#=
2D single-site (square/rectangular) lattice, for standalone tests of the general
Jastrow. Site (x,y) → index (x-1) + (y-1)*Lx + 1.
=#
function Lattice2D(Lx::Int, Ly::Int; periodic::Bool=true)
    L = Lx * Ly
    site(x, y) = (mod1(x, Lx) - 1) + (mod1(y, Ly) - 1) * Lx + 1
    neighbors = [Int[] for _ in 1:L]
    for y in 1:Ly, x in 1:Lx
        i = site(x, y)
        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            x2, y2 = x + dx, y + dy
            (periodic || (1 ≤ x2 ≤ Lx && 1 ≤ y2 ≤ Ly)) || continue
            j = site(x2, y2)
            j in neighbors[i] || push!(neighbors[i], j)
            i in neighbors[j] || push!(neighbors[j], i)
        end
    end
    cells = Matrix{Int}(undef, 2, L)
    for y in 1:Ly, x in 1:Lx
        cells[:, site(x, y)] .= (x - 1, y - 1)
    end
    basis  = zeros(Int, L)
    period = periodic ? [Lx, Ly] : [0, 0]
    return GeneralLattice(neighbors, cells, basis, period)
end
