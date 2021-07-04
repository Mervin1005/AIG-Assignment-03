### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 6790f345-96a1-45d0-ab6e-a97d533d4b9f
using Pkg

# ╔═╡ 8fbc3668-bbda-4691-bcb1-a2f1b56df894
using PlutoUI

# ╔═╡ a8b7d309-c42c-4c6c-96ca-1a1bc4a63f41
using Markdown, InteractiveUtils, Combinatorics, Parameters, MathOptInterface, QuantEcon, LinearAlgebra, Polyhedra,Clp,  Random

# ╔═╡ ccc87fbc-a12a-462f-ae2a-2f7a7064b9c8
using LinearAlgebra: LAPACKException, SingularException

# ╔═╡ 4e240dd4-3511-4129-bf8a-028abd208f70
using QuantEcon: next_k_array!

# ╔═╡ 492e6b5a-b237-4abd-8b4a-967bb708d58b
md"####### captures sequential game between players in normal form and computes payoff of players with mixed strategy"

# ╔═╡ 65b0cf39-f5ce-40d6-9ecc-6954a9b83b49
begin
	struct Player{N,T<:Real}
    payoff_array::Array{T,N}

    function Player{N,T}(payoff_array::Array{T,N}) where {N,T<:Real}
        if prod(size(payoff_array)) == 0
            throw(ArgumentError("each player must at least have one action"))
        end
        return new(payoff_array)
    end
end
	
	Player(payoff_array::Array{T,N}) where {T<:Real,N} = Player{N,T}(payoff_array)
	
	Player{N,T}(player::Player{N,S}) where {N,T,S} = Player(Array{T}(player.payoff_array))
	
	Base.convert(::Type{T}, player::Player) where {T<:Player} = player isa T ? player : T(player)# instance of new player conversion
	
	Player(::Type{T}, player::Player{N}) where {T<:Real,N} = Player{N,T}(player)

    Player(player::Player{N,T}) where {N,T} = Player{N,T}(player)

    num_actions(p::Player) = size(p.payoff_array, 1)
    num_opponents(::Player{N}) where {N} = N - 1
end

# ╔═╡ af2410b6-d3c4-4e11-94e0-7cb406d3abaf
md"###### Pure and Mixed Strategy Evaluation  methods and constructors "

# ╔═╡ d47bf9b3-8708-49ae-8b1c-cdfaaa6e5109
const MOI = MathOptInterface; const MOIU = MOI.Utilities; const PureAction = Integer

# ╔═╡ cf973769-b603-43cd-bc60-55bbfda906a4
MixedAction{T<:Real} = Vector{T}

# ╔═╡ c2817a51-db62-4d41-b97d-5f08e0bce85b
Action{T<:Real} = Union{PureAction,MixedAction{T}}

# ╔═╡ 44ddc997-1a89-4c8c-aa94-b4e760ddc43d
PureActionProfile{N,T<:PureAction} = NTuple{N,T}

# ╔═╡ 241b14c1-ae05-414b-92c9-575b97a0b55d
MixedActionProfile{T<:Real,N} = NTuple{N,MixedAction{T}}

# ╔═╡ c1c8892e-d1ba-4280-bcee-0cdd541552ea
const ActionProfile = Union{PureActionProfile,MixedActionProfile}

# ╔═╡ e2f229e3-5686-4d8f-b3cb-5e7bb71d3ce0
Base.summary(player::Player) =
    string(Base.dims2string(size(player.payoff_array)),
           " ",
           split(string(typeof(player)), ".")[end])

# ╔═╡ 6f26b807-25da-49e1-8914-a5d777e98faf
function Base.show(io::IO, player::Player)
    print(io, summary(player))
    println(io, ":")
    Base.print_array(io, player.payoff_array)
end

# ╔═╡ d6f2ea61-70d4-4231-b35d-aafd0cc90406
function payoff_vector(player::Player, opponents_actions::Tuple{})
    throw(ArgumentError("input tuple must not be empty"))
end

# ╔═╡ 5d035d9d-91b7-4668-ac55-a0ca18b4cd9b
function payoff_vector(player::Player{N,T1},
                       opponents_actions::MixedActionProfile{T2}) where {N,T1,T2}
    length(opponents_actions) != num_opponents(player) &&
        throw(ArgumentError(
            "length of opponents_actions must be $(num_opponents(player))"
        ))
    S = promote_type(T1, T2)
    payoffs::Array{S,N} = player.payoff_array
    for i in num_opponents(player):-1:1
        payoffs = _reduce_ith_opponent(payoffs, i, opponents_actions[i])
    end
    return vec(payoffs)
end

# ╔═╡ 627bd43c-2561-4bb7-85aa-f9badfbc26b3
function payoff_vector(player::Player{2}, opponent_action::MixedAction)
    return player.payoff_array * opponent_action
end

# ╔═╡ 244b3130-d981-40cc-9b6d-1b6296bd110c
function payoff_vector(player::Player{1}, opponent_action::Nothing)
    return player.payoff_array
end

# ╔═╡ d495f3ec-23b6-4cc6-8d5c-a5b7dee50131
md"####### Mixed Strategy Function Definitions"

# ╔═╡ 613d63f6-8ff8-4ed8-ada9-76512b43827b
#is_best_response return true if `own_action` is a best response to `opponents_actionss`.
function is_best_response(player::Player,
                          own_action::MixedAction,
                          opponents_actions::Union{Action,ActionProfile,Nothing};
                          tol::Real=1e-8)
    payoffs = payoff_vector(player, opponents_actions)
    payoff_max = maximum(payoffs)
    return dot(own_action, payoffs) >= payoff_max - tol
end

# ╔═╡ 30eb21c5-95fd-4a0c-b113-05878a6b957b
#Return all the best response actions to `opponents_action
function best_responses(player::Player,
                        opponents_actions::Union{Action,ActionProfile,Nothing};
                        tol::Real=1e-8)
    payoffs = payoff_vector(player, opponents_actions)
    payoff_max = maximum(payoffs)
    best_responses = findall(x -> x >= payoff_max - tol, payoffs)
    return best_responses
end

# ╔═╡ 47f82fa2-c385-46ae-99a2-ea61d57bc30b
# Return a best response action to `opponents_actions
function best_response(rng::AbstractRNG, player::Player,
                       opponents_actions::Union{Action,ActionProfile,Nothing};
                       tie_breaking::Symbol=:smallest,
                       tol::Real=1e-8)
    if tie_breaking == :smallest
        payoffs = payoff_vector(player, opponents_actions)
        return argmax(payoffs)
    elseif tie_breaking == :random
        brs = best_responses(player, opponents_actions; tol=tol)
        return rand(rng, brs)
    else
        throw(ArgumentError(
            "tie_breaking must be one of `:smallest` or `:random`"
        ))
    end
end

# ╔═╡ 372cb136-8cc9-4ed8-9ebb-0d4dd1388c1e
best_response(player::Player,
              opponents_actions::Union{Action,ActionProfile,Nothing};
              tie_breaking::Symbol=:smallest, tol::Real=1e-8) =
    best_response(Random.GLOBAL_RNG, player, opponents_actions,
                  tie_breaking=tie_breaking, tol=tol)

# ╔═╡ 8e6f04ba-ed6f-4150-8ef2-1e0a4e5c76d7
#Struct to contain options for `best_response`.
@with_kw struct Options{T<:Real,TR<:AbstractRNG}
    tol::T = 1e-8
    tie_breaking::Symbol = :smallest
    rng::TR = Random.GLOBAL_RNG
end

# ╔═╡ e8f5be25-b003-421a-b862-f75969d3316d
#Return a best response action to `opponents_actions` with options as specified by a `BROptions` instance `options`."""
best_response(player::Player,
              opponents_actions::Union{Action,ActionProfile,Nothing},
              options::Options) =
    best_response(options.rng, player, opponents_actions;
                  tie_breaking=options.tie_breaking, tol=options.tol)

# ╔═╡ 1499f0cf-af6d-4f8e-ab02-191906a74e28
function best_response(player::Player,
                       opponents_actions::Union{Action,ActionProfile,Nothing},
                       payoff_perturbation::Vector{Float64})
    length(payoff_perturbation) != num_actions(player) &&
        throw(ArgumentError(
            "length of payoff_perturbation must be $(num_actions(player))"
        ))

    payoffs = payoff_vector(player, opponents_actions) + payoff_perturbation
    return argmax(payoffs)
end

# ╔═╡ ff451d14-8086-4e54-9838-5a6c62ccf6be
# Check that the shapes of the payoff arrays are consistent
function is_consistent(players::NTuple{N,Player{N,T}}) where {N,T}
    shape_1 = size(players[1].payoff_array)
    for i in 2:N
        shape = size(players[i].payoff_array)
        for j in 1:(N-i+1)
            shape[j] == shape_1[i-1+j] || return false
        end
        for j in (N-i+2):N
            shape[j] == shape_1[j-(N-i+1)] || return false
        end
    end
    return true
	
	
end

# ╔═╡ 540bb8d8-094e-4c80-937e-1726bec94669
begin
	struct NormalFormGame{N,T<:Real}
	    players::NTuple{N,Player{N,T}}
	    nums_actions::NTuple{N,Int}
	end
	
	num_players(::NormalFormGame{N}) where {N} = N
	
	function NormalFormGame(::Tuple{})  # To resolve definition ambiguity
	    throw(ArgumentError("input tuple must not be empty"))
	end
	
	function NormalFormGame(T::Type, nums_actions::NTuple{N,Int}) where N
    players = Vector{Player{N,T}}(undef, N)
    for i in 1:N
        sz = ntuple(j -> nums_actions[i-1+j <= N ? i-1+j : i-1+j-N], N)
        players[i] = Player(zeros(T, sz))
    end
    return NormalFormGame(players)
    end
	
	NormalFormGame(nums_actions::NTuple{N,Int}) where {N} =
    NormalFormGame(Float64, nums_actions)
	
	# Constructor of an N-player NormalFormGame with a tuple of N Player instances.
	function NormalFormGame(players::NTuple{N,Player{N,T}}) where {N,T}
    is_consistent(players) ||
        throw(ArgumentError("shapes of payoff arrays must be consistent"))
    nums_actions = ntuple(i -> num_actions(players[i]), N)
    return NormalFormGame{N,T}(players, nums_actions)
    end
	
	#Constructor of an N-player NormalFormGame with a vector of N Player instances.
    NormalFormGame(players::Vector{Player{N,T}}) where {N,T} =
    NormalFormGame(ntuple(i -> players[i], N))
	
	# Constructor of an N-player NormalFormGame with N Player instances.
	function NormalFormGame(players::Player{N,T}...) where {N,T}
    length(players) != N && error("Need $N players")
    NormalFormGame(players)  # use constructor for Tuple of players above
    end
	
	#NormalFormGame(payoffs)Construct an N-player NormalFormGame for N>=2 with an   array `payoffs` of M=N+1 dimensions, where `payoffs[a_1, a_2, ..., a_N, :]` contains a profile of N payoff values
	function NormalFormGame(payoffs::Array{T,M}) where {T<:Real,M}
    N = M - 1
    dims = Base.front(size(payoffs))
    colons = Base.front(ntuple(j -> Colon(), M))

    size(payoffs)[end] != N && throw(ArgumentError(
        "length of the array in the last axis must be equal to
         the number of players"
    ))

    players = ntuple(
        i -> Player(permutedims(view(payoffs, colons..., i),
                                     (i:N..., 1:i-1...)::typeof(dims))
                   ),
        Val(N)
    )
    NormalFormGame(players)
    end
	
	# NormalFormGame(payoffs) Construct a symmetric 2-player NormalFormGame with a square matrix.
	function NormalFormGame(payoffs::Matrix{T}) where T<:Real
    n, m = size(payoffs)
    n != m && throw(ArgumentError(
        "symmetric two-player game must be represented by a square matrix"
    ))
    player = Player(payoffs)
    return NormalFormGame(player, player)
    end
	
	function NormalFormGame{N,T}(g::NormalFormGame{N,S}) where {N,T,S}
    players_new = ntuple(i -> Player{N,T}(g.players[i]), Val(N))
    return NormalFormGame(players_new)
    end

    NormalFormGame(::Type{T}, g::NormalFormGame{N}) where {T<:Real,N} =
    NormalFormGame{N,T}(g)

    NormalFormGame(g::NormalFormGame{N,T}) where {N,T} = NormalFormGame{N,T}(g)


end

# ╔═╡ f8195d53-4be6-4177-a38e-52f1036453c6
md"###### delete_action  delete_action(g, action, player_idx) Return a new  `NormalFormGame` instance with the action(s) specified by `action` deleted from the action set of the player specified by `player_idx`.
	 delete action for pure action"

# ╔═╡ 37e38568-5157-420b-98b7-9420007f4dad
begin
	function delete_action(player::Player{N,T}, action::AbstractVector{<:PureAction},
                       player_idx::Integer=1) where {N,T}
    sel = Any[Colon() for i in 1:N]
    sel[player_idx] = setdiff(axes(player.payoff_array, player_idx), action)
    payoff_array_new = player.payoff_array[sel...]::Array{T,N}
    return Player(payoff_array_new)
    end

delete_action(player::Player, action::PureAction, player_idx::Integer=1) =
    delete_action(player, [action], player_idx) 
	
	
	function delete_action(g::NormalFormGame{N},
                       action::AbstractVector{<:PureAction},
                       player_idx::Integer) where N
    players_new  = [delete_action(player, action,
                    player_idx-i+1>0 ? player_idx-i+1 : player_idx-i+1+N)
                    for (i, player) in enumerate(g.players)]
    return NormalFormGame(players_new)
    end

    delete_action(g::NormalFormGame, action::PureAction, player_idx::Integer) =
    delete_action(g, [action], player_idx)
end

# ╔═╡ 5c219df7-9bc0-4e28-a769-d828eac591ab
#  NormalFormGame(T, g) Convert `g` into a new `NormalFormGame` instance with eltype `T`.
Base.convert(::Type{T}, g::NormalFormGame) where {T<:NormalFormGame} =
    g isa T ? g : T(g)

# ╔═╡ 68b33f67-c38f-4892-99e1-c90c94ebc4fe
Base.summary(g::NormalFormGame) =
    string(Base.dims2string(g.nums_actions),
           " ",
           split(string(typeof(g)), ".")[end])

# ╔═╡ 03c0aa3f-29c1-4187-a6ec-ad7f2f695d95
md"########TODO: add printout of payoff arrays"

# ╔═╡ 66e92fe3-715c-4586-88fc-9c33093d35a7
function Base.show(io::IO, g::NormalFormGame)
    print(io, summary(g))
end

# ╔═╡ b7418965-5a7d-4f95-8236-7ab60dfd0d4c
md"##### Player Action value index definition"

# ╔═╡ 1d9d8f13-d1ac-4280-a160-f5aa3019f3d9
begin
	function Base.getindex(g::NormalFormGame{N,T},
	                       index::Integer...) where {N,T}
	    length(index) != N &&
	        throw(DimensionMismatch("index must be of length $N"))
	
	    payoff_profile = Array{T}(undef, N)
	    for (i, player) in enumerate(g.players)
	        payoff_profile[i] =
	            player.payoff_array[(index[i:end]..., index[1:i-1]...)...]
	    end
	    return payoff_profile
	end
	
	function Base.getindex(g::NormalFormGame{1,T}, index::Integer) where T
	    return g.players[1].payoff_array[index]
	end
	Base.getindex(g::NormalFormGame{N}, ci::CartesianIndex{N}) where {N} =
	    g[to_indices(g, (ci,))...]
end

# ╔═╡ 1dd75a0c-9ab2-405b-a5ff-f9a55c53ff09
begin
	function Base.setindex!(g::NormalFormGame{N,T},
	                        payoff_profile::Vector{S},
	                        index::Integer...) where {N,T,S<:Real}
	    length(index) != N &&
	        throw(DimensionMismatch("index must be of length $N"))
	    length(payoff_profile) != N &&
	        throw(DimensionMismatch("assignment must be of $N-array"))
	
	    for (i, player) in enumerate(g.players)
	        player.payoff_array[(index[i:end]...,
	                             index[1:i-1]...)...] = payoff_profile[i]
	    end
	    return payoff_profile
	end
	
	# Trivial game with 1 player
	function Base.setindex!(g::NormalFormGame{1,T},
	                        payoff::S,
	                        index::Integer) where {T,S<:Real}
	    g.players[1].payoff_array[index] = payoff
	    return payoff
	end
	
	Base.setindex!(g::NormalFormGame{N}, v, ci::CartesianIndex{N}) where {N} =
	    g[to_indices(g, (ci,))...] = v
	
end

# ╔═╡ 77b7f2ce-9031-4ef9-8f50-dc8713c60726
md"#### Nash Equilibrium"

# ╔═╡ c6e90d98-70a4-4708-8ec3-17c847ae012e
begin
	# is_nash
	
	function is_nash(g::NormalFormGame, action_profile::ActionProfile;
	                 tol::Real=1e-8)
	    for (i, player) in enumerate(g.players)
	        own_action = action_profile[i]
	        opponents_actions =
	            tuple(action_profile[i+1:end]..., action_profile[1:i-1]...)
	        if !(is_best_response(player, own_action, opponents_actions, tol=tol))
	            return false
	        end
	    end
	    return true
	end
	
	
	function is_nash(g::NormalFormGame{2}, action_profile::ActionProfile;
	                 tol::Real=1e-8)
	    for (i, player) in enumerate(g.players)
	        own_action, opponent_action =
	            action_profile[i], action_profile[3-i]
	        if !(is_best_response(player, own_action, opponent_action, tol=tol))
	            return false
	        end
	    end
	    return true
	end
	
	is_nash(g::NormalFormGame{1}, action::Action; tol::Real=1e-8) =
	    is_best_response(g.players[1], action, nothing, tol=tol)
	
	is_nash(g::NormalFormGame{1}, action_profile::ActionProfile;
	        tol::Real=1e-8) = is_nash(g, action_profile..., tol=tol)
end

# ╔═╡ 4ce9ef7e-c859-4b6e-9e09-ee3461925dfa
#Convert a pure action to the corresponding mixed action.
function pure2mixed(num_actions::Integer, action::PureAction)
    mixed_action = zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action
end

# ╔═╡ 3df4879b-9a85-4eab-94ec-3ee8b009af70
begin
	function is_dominated(
	    ::Type{T}, player::Player{1}, action::PureAction; tol::Real=1e-8,
	    lp_solver::Union{Type{TO},Function}=() -> Clp.Optimizer(LogLevel=0)
	) where {T<:Real,TO<:MOI.AbstractOptimizer}
	        payoff_array = player.payoff_array
	        return maximum(payoff_array) > payoff_array[action] + tol
	end
	
	is_dominated(
	    player::Player, action::PureAction; tol::Real=1e-8,
	    lp_solver::Union{Type{TO},Function}=() -> Clp.Optimizer(LogLevel=0)
	) where {TO<:MOI.AbstractOptimizer} =
	    is_dominated(Float64, player, action, tol=tol, lp_solver=lp_solver)
	
end

# ╔═╡ 73f96d6d-2367-4ad3-b63a-4913b317ed89
begin
	# Return a vector of actions that are strictly dominated by some mixed actions.
	function dominated_actions(
	    ::Type{T}, player::Player; tol::Real=1e-8,
	    lp_solver::Union{Type{TO},Function}=() -> Clp.Optimizer(LogLevel=0)
	) where {T<:Real,TO<:MOI.AbstractOptimizer}
	    out = Int[]
	    for action = 1:num_actions(player)
	        if is_dominated(T, player, action, tol=tol, lp_solver=lp_solver)
	            append!(out, action);
	        end
	    end
	
	    return out
	end
	
	dominated_actions(
	    player::Player; tol::Real=1e-8,
	    lp_solver::Union{Type{TO},Function}=() -> Clp.Optimizer(LogLevel=0)
	) where {TO<:MOI.AbstractOptimizer} =
	    dominated_actions(Float64, player, tol=tol, lp_solver=lp_solver)
	
end

# ╔═╡ 1cbf020b-289c-4833-90fc-614fc1f4d936
begin
	# Return a tuple of random mixed actions (vectors of floats)
	random_mixed_actions(rng::AbstractRange, nums_actions::NTuple{N,Int}) where {N} =
	    ntuple(i -> QuantEcon.random_probvec(rng, nums_actions[i]), Val(N))
	
	random_mixed_actions(nums_actions::NTuple{N,Int}) where {N} =
	    random_mixed_actions(Random.GLOBAL_RNG, nums_actions)
	
end

# ╔═╡ b5955180-d95a-4c62-8e36-3aec1d953b91
md"### Policy Solver"

# ╔═╡ 617e6543-6819-4ae9-8d47-c22f0af1f540
begin
	function _solve!(A::Matrix{T}, b::Vector{T}) where T <: Union{Float64,Float32}
	    r = 0
	    try
	        LAPACK.gesv!(A, b)
	    catch LAPACKException
	        r = 1
	    end
	    return r
	end
	
	@inline function _solve!(A::Matrix{Rational{T}},
	                         b::Vector{Rational{T}}) where T <: Integer
	    r = 0
	    try
	        b[:] = ldiv!(lu!(A), b)
	    catch SingularException
	        r = 1
	    end
	    return r
	end
end

# ╔═╡ 3afe8191-1ec5-4465-b7c7-021950eec743
coin_toss_bimatrix = Array{Float64}(undef, 2, 2, 2)

# ╔═╡ d0e15eae-6cbb-4cca-bff6-ec8c41619b08
coin_toss_bimatrix[1, 1, :] = [1, -1]  # payoff profile for action profile (1, 1)

# ╔═╡ 88581ac0-e411-4576-85ec-09a1a569b105
coin_toss_bimatrix[1, 2, :] = [-1, 1]

# ╔═╡ 8c9fdaa5-9460-465f-bbe8-8077ca0443ce
coin_toss_bimatrix[2, 1, :] = [-1, 1]

# ╔═╡ e646836a-987e-4b17-a464-45f8ce016c1b
coin_toss_bimatrix[2, 2, :] = [1, -1]

# ╔═╡ 013f9648-339e-4b10-b69c-f15dda810a6b
g_CoinToss = NormalFormGame(coin_toss_bimatrix)

# ╔═╡ 322e8fae-41ce-4535-8381-60994c102501
g_CoinToss.players[1]  # Player instance for player 1

# ╔═╡ 93988755-8f0e-4ab8-b594-9faa6d002c05
g_CoinToss.players[2]  # Player instance for player 2

# ╔═╡ c5ca7f67-5993-446f-b750-9e2292229b5e
md"### Player's Payoff Matrix"

# ╔═╡ 7a24bd4f-ad59-40a9-9866-a85d0f9e35bc
g_CoinToss.players[1].payoff_array

# ╔═╡ a0a6f792-3067-411c-83c1-3cd5babf95c3
g_CoinToss.players[2].payoff_array

# ╔═╡ 8ccae1c8-2f3f-43ce-88ee-5ccc0bd456f4
g_CoinToss[1, 1]  # payoff profile for action profile (1, 1)

# ╔═╡ 42099ff0-f0c3-4de9-b83e-1f8a6afd4e3d
variant_watch_matrix = [4 0;
                            3 2]  # square matrix

# ╔═╡ 0ac28bbd-2d98-49c0-82f2-d49e8d2a2b8f
g_TVA = NormalFormGame(variant_watch_matrix)

# ╔═╡ 5a0b3049-1b34-4d2b-84a5-fb16bccb177a
g_TVA.players[2].payoff_array  # Player 2's payoff array

# ╔═╡ 5bee395f-b989-4fdb-9208-ceee8d8a89b6
RockPapSza_matrix = [0 -1 1;
              1 0 -1;
              -1 1 0]

# ╔═╡ 56d512fc-7a93-4ce4-ad1c-e10430f5b985
g_RockPapzaS = NormalFormGame(RockPapSza_matrix)

# ╔═╡ 246fab84-b751-4bff-8a32-ea412c800613
md"### Creating Players"

# ╔═╡ 205802c2-2c8a-449c-a809-aae9d7c98f48
# A final way to create a norm form game is to pass an array of dimensions that represent player's payoff

# ╔═╡ 940a160c-d88e-4ece-a2f9-93f65342b483
md"### Gender Wars Game"

# ╔═╡ ef985371-d89a-4005-b44f-d00969b7e9de
playerM = Player([3 1; 0 2])

# ╔═╡ b5844113-7831-491d-8cd8-54828f2736a3
playerF = Player([2 0; 1 3])

# ╔═╡ 5d643eac-bf06-4df0-a759-d278876d6529
playerM.payoff_array

# ╔═╡ 3a2d7ea7-5822-4acd-b6f2-edd5293bcc26
playerF.payoff_array

# ╔═╡ d7c67861-dfc1-4933-8dc8-b8e9cd2f9c3e
# Passing an array of Player instances is another way to create a NormalFormGame instance:
game_GenWars = NormalFormGame((playerM, playerF))

# ╔═╡ e6593a0c-6a69-4a26-adf8-80e3ef7e1ea7
a, c = 80, 20

# ╔═╡ bcbc274c-7f07-4f09-8e13-d84f722cce83
N = 3

# ╔═╡ ed5f44b6-3c55-4baa-a1bf-5a06fc57fcab
q_grid_size = 13

# ╔═╡ b7528443-345a-4aee-8649-9bf7a7e6b885
q_grid = range(0, step=div(a-c, q_grid_size-1), length=q_grid_size)  # [0, 5, 10, ..., 60]

# ╔═╡ Cell order:
# ╠═6790f345-96a1-45d0-ab6e-a97d533d4b9f
# ╠═8fbc3668-bbda-4691-bcb1-a2f1b56df894
# ╠═a8b7d309-c42c-4c6c-96ca-1a1bc4a63f41
# ╠═ccc87fbc-a12a-462f-ae2a-2f7a7064b9c8
# ╠═4e240dd4-3511-4129-bf8a-028abd208f70
# ╠═492e6b5a-b237-4abd-8b4a-967bb708d58b
# ╠═65b0cf39-f5ce-40d6-9ecc-6954a9b83b49
# ╠═af2410b6-d3c4-4e11-94e0-7cb406d3abaf
# ╠═d47bf9b3-8708-49ae-8b1c-cdfaaa6e5109
# ╠═cf973769-b603-43cd-bc60-55bbfda906a4
# ╠═c2817a51-db62-4d41-b97d-5f08e0bce85b
# ╠═44ddc997-1a89-4c8c-aa94-b4e760ddc43d
# ╠═241b14c1-ae05-414b-92c9-575b97a0b55d
# ╠═c1c8892e-d1ba-4280-bcee-0cdd541552ea
# ╠═e2f229e3-5686-4d8f-b3cb-5e7bb71d3ce0
# ╠═6f26b807-25da-49e1-8914-a5d777e98faf
# ╠═d6f2ea61-70d4-4231-b35d-aafd0cc90406
# ╠═5d035d9d-91b7-4668-ac55-a0ca18b4cd9b
# ╠═627bd43c-2561-4bb7-85aa-f9badfbc26b3
# ╠═244b3130-d981-40cc-9b6d-1b6296bd110c
# ╠═d495f3ec-23b6-4cc6-8d5c-a5b7dee50131
# ╠═613d63f6-8ff8-4ed8-ada9-76512b43827b
# ╠═30eb21c5-95fd-4a0c-b113-05878a6b957b
# ╠═47f82fa2-c385-46ae-99a2-ea61d57bc30b
# ╠═372cb136-8cc9-4ed8-9ebb-0d4dd1388c1e
# ╠═8e6f04ba-ed6f-4150-8ef2-1e0a4e5c76d7
# ╠═e8f5be25-b003-421a-b862-f75969d3316d
# ╠═1499f0cf-af6d-4f8e-ab02-191906a74e28
# ╠═ff451d14-8086-4e54-9838-5a6c62ccf6be
# ╠═540bb8d8-094e-4c80-937e-1726bec94669
# ╠═f8195d53-4be6-4177-a38e-52f1036453c6
# ╠═37e38568-5157-420b-98b7-9420007f4dad
# ╠═5c219df7-9bc0-4e28-a769-d828eac591ab
# ╠═68b33f67-c38f-4892-99e1-c90c94ebc4fe
# ╠═03c0aa3f-29c1-4187-a6ec-ad7f2f695d95
# ╠═66e92fe3-715c-4586-88fc-9c33093d35a7
# ╠═b7418965-5a7d-4f95-8236-7ab60dfd0d4c
# ╠═1d9d8f13-d1ac-4280-a160-f5aa3019f3d9
# ╠═1dd75a0c-9ab2-405b-a5ff-f9a55c53ff09
# ╠═77b7f2ce-9031-4ef9-8f50-dc8713c60726
# ╠═c6e90d98-70a4-4708-8ec3-17c847ae012e
# ╠═4ce9ef7e-c859-4b6e-9e09-ee3461925dfa
# ╠═3df4879b-9a85-4eab-94ec-3ee8b009af70
# ╠═73f96d6d-2367-4ad3-b63a-4913b317ed89
# ╠═1cbf020b-289c-4833-90fc-614fc1f4d936
# ╠═b5955180-d95a-4c62-8e36-3aec1d953b91
# ╠═617e6543-6819-4ae9-8d47-c22f0af1f540
# ╠═3afe8191-1ec5-4465-b7c7-021950eec743
# ╠═d0e15eae-6cbb-4cca-bff6-ec8c41619b08
# ╠═88581ac0-e411-4576-85ec-09a1a569b105
# ╠═8c9fdaa5-9460-465f-bbe8-8077ca0443ce
# ╠═e646836a-987e-4b17-a464-45f8ce016c1b
# ╠═013f9648-339e-4b10-b69c-f15dda810a6b
# ╠═322e8fae-41ce-4535-8381-60994c102501
# ╠═93988755-8f0e-4ab8-b594-9faa6d002c05
# ╠═c5ca7f67-5993-446f-b750-9e2292229b5e
# ╠═7a24bd4f-ad59-40a9-9866-a85d0f9e35bc
# ╠═a0a6f792-3067-411c-83c1-3cd5babf95c3
# ╠═8ccae1c8-2f3f-43ce-88ee-5ccc0bd456f4
# ╠═42099ff0-f0c3-4de9-b83e-1f8a6afd4e3d
# ╠═0ac28bbd-2d98-49c0-82f2-d49e8d2a2b8f
# ╠═5a0b3049-1b34-4d2b-84a5-fb16bccb177a
# ╠═5bee395f-b989-4fdb-9208-ceee8d8a89b6
# ╠═56d512fc-7a93-4ce4-ad1c-e10430f5b985
# ╠═246fab84-b751-4bff-8a32-ea412c800613
# ╠═205802c2-2c8a-449c-a809-aae9d7c98f48
# ╠═940a160c-d88e-4ece-a2f9-93f65342b483
# ╠═ef985371-d89a-4005-b44f-d00969b7e9de
# ╠═b5844113-7831-491d-8cd8-54828f2736a3
# ╠═5d643eac-bf06-4df0-a759-d278876d6529
# ╠═3a2d7ea7-5822-4acd-b6f2-edd5293bcc26
# ╠═d7c67861-dfc1-4933-8dc8-b8e9cd2f9c3e
# ╠═e6593a0c-6a69-4a26-adf8-80e3ef7e1ea7
# ╠═bcbc274c-7f07-4f09-8e13-d84f722cce83
# ╠═ed5f44b6-3c55-4baa-a1bf-5a06fc57fcab
# ╠═b7528443-345a-4aee-8649-9bf7a7e6b885
