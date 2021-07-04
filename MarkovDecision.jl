### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ c70629cd-89a6-4e8f-addf-3abaeeac6041
using Pkg

# ╔═╡ 520c2c98-a170-4563-abba-67120d7ccaa3
Pkg.add("POMDPPolicies")

# ╔═╡ f176a3d8-c386-43cc-a1c2-d5f18af0ffd0
using Markdown

# ╔═╡ 505b8132-4347-4e50-9e6c-3dded1b360ad
using InteractiveUtils

# ╔═╡ 02d7619e-39fb-4681-82b6-3f3b90f756f1
using POMDPs

# ╔═╡ 8081d7a0-8bf1-421c-a633-0b7252ad2405
using POMDPModelTools

# ╔═╡ ae703b16-934e-49d9-bdf5-bde468f021be
using POMDPPolicies

# ╔═╡ e10f5906-886b-432e-aa87-faddb1290681
using POMDPSimulators

# ╔═╡ f36eab1d-075a-452a-9d29-92d8d7856608
using DiscreteValueIteration

# ╔═╡ 2b91da86-c959-4fc4-b492-fdd1ef347b27
struct GridState
	x::Int64
	y::Int64
	done::Bool
end

# ╔═╡ 8c7c19d1-80b7-4749-b210-0d347a1a4744
GS(x::Int64, y::Int64) = GridState(x,y,false)

# ╔═╡ 380d831e-8237-4bf7-82e6-c94c46ea0765
posequal(s1::GridState, s2::GridState) = s1.x == s2.x && s1.y == s2.y

# ╔═╡ a4569e38-0047-4be2-b53f-bc2558f4996c
mutable struct GridStruct <: MDP{GridState, Symbol}
    xaxis::Int64
    yaxis::Int64
    reward_states::Vector{GridState}
    reward_values::Vector{Float64}
    transprob::Float64
    discount_factor::Float64
end

# ╔═╡ 1e0a2e2f-ec09-4926-af10-57e320b2b213
function GW(;sx::Int64=5,
                    sy::Int64=5,
                    rs::Vector{GridState}=[GS(1,3), GS(1,4), GS(3,1),GS(3,4),GS(3,5),GS(4,2),GS(4,5),GS(5,3),GS(5,5)],
                    rv::Vector{Float64}=rv = [-10.,3,-3,3,-3,7,3,-3,10],
                    tp::Float64=0.7,
                    discount_factor::Float64=0.9)
    return GridStruct(sx, sy, rs, rv, tp, discount_factor)
end

# ╔═╡ 17861ad7-04ee-45c6-a310-51da2009b865
function POMDPs.states(mdp::GridStruct)
    s = GridState[]
    for d = 0:1, y = 1:mdp.yaxis, x = 1:mdp.xaxis
        push!(s, GridState(x,y,d))
    end
    return s
end;

# ╔═╡ 9e3332c2-12ae-47df-8b40-61442cb4b479
POMDPs.actions(mdp::GridStruct) = [:up, :down, :left, :right];

# ╔═╡ d7ccf03a-835a-472b-914c-5af5f7dc3872
function inbounds(mdp::GridStruct,x::Int64,y::Int64)
    if 1 <= x <= mdp.xaxis && 1 <= y <= mdp.yaxis
        return true
    else
        return false
    end
end

# ╔═╡ 77ad7e4e-546b-4402-a39b-78223fd52af9
inbounds(mdp::GridStruct, state::GridState) = inbounds(mdp, state.x, state.y);

# ╔═╡ 0cd05a44-b82e-4217-a0f4-8b6d3167d936
function POMDPs.transition(mdp::GridStruct, state::GridState, action::Symbol)
    a = action
    x = state.x
    y = state.y
    
    if state.done
        return SparseCat([GridState(x, y, true)], [1.0])
    elseif state in mdp.reward_states
        return SparseCat([GridState(x, y, true)], [1.0])
    end

    neighbors = [
        GridState(x+1, y, false),
        GridState(x-1, y, false),
        GridState(x, y-1, false),
        GridState(x, y+1, false),
        ]
    
    targets = Dict(:right=>1, :left=>2, :down=>3, :up=>4)
    target = targets[a]
    
    probability = fill(0.0, 4)

    if !inbounds(mdp, neighbors[target])
        return SparseCat([GS(x, y)], [1.0])
    else
        probability[target] = mdp.transprob

        oob_count = sum(!inbounds(mdp, n) for n in neighbors) 

        new_probability = (1.0 - mdp.transprob)/(3-oob_count)

        for i = 1:4
            if inbounds(mdp, neighbors[i]) && i != target
                probability[i] = new_probability
            end
        end
    end

    return SparseCat(neighbors, probability)
end;

# ╔═╡ 79c03808-bbb3-424f-b678-5858b0fd2d73
function POMDPs.reward(mdp::GridStruct, state::GridState, action::Symbol, statep::GridState) #deleted action
    if state.done
        return 0.0
    end
    r = 0.0
    n = length(mdp.reward_states)
    for i = 1:n
        if posequal(state, mdp.reward_states[i])
            r += mdp.reward_values[i]
        end
    end
    return r
end;

# ╔═╡ 8e5d760d-4a3c-4326-9e2d-e4b3918bb8c0
POMDPs.discount(mdp::GridStruct) = mdp.discount_factor;

# ╔═╡ 77b549ac-6204-455d-84b7-5a6653bc89e6
function POMDPs.stateindex(mdp::GridStruct, state::GridState)
    sd = Int(state.done + 1)
    ci = CartesianIndices((mdp.xaxis, mdp.yaxis, 2))
    return LinearIndices(ci)[state.x, state.y, sd]
end

# ╔═╡ 658e1360-8604-42f9-9aea-f39c461ed5f5
function POMDPs.actionindex(mdp::GridStruct, act::Symbol)
    if act==:up
        return 1
    elseif act==:down
        return 2
    elseif act==:left
        return 3
    elseif act==:right
        return 4
    end
    error("Invalid GridStruct action: $act")
end;

# ╔═╡ 82243cb7-f460-4407-ac55-669402a6db5e
POMDPs.isterminal(mdp::GridStruct, s::GridState) = s.done

# ╔═╡ ab89293f-725c-4ff9-b683-01a84652e0c7
POMDPs.initialstate(pomdp::GridStruct) = Deterministic(GS(1,1))

# ╔═╡ 4311478b-0b1e-4939-bf87-63b64e1c70b1
mdp = GW()

# ╔═╡ 04a8aa2e-58e0-46d0-8c61-7fe00b5cc176
solver = ValueIterationSolver(max_iterations=100, belres=1e-3; verbose=true)

# ╔═╡ ce9c3817-ae4c-4419-bda9-aa3b64eba3ce
policy = solve(solver, mdp);

# ╔═╡ 62d5d697-bcbd-4b8a-9d6c-a41907cbfc79
for (s,a,r) in stepthrough(mdp, policy, "s,a,r", max_steps=20)
    @show s
    @show a
    @show r
    println(s, a, r)
end

# ╔═╡ Cell order:
# ╠═f176a3d8-c386-43cc-a1c2-d5f18af0ffd0
# ╠═c70629cd-89a6-4e8f-addf-3abaeeac6041
# ╠═505b8132-4347-4e50-9e6c-3dded1b360ad
# ╠═02d7619e-39fb-4681-82b6-3f3b90f756f1
# ╠═8081d7a0-8bf1-421c-a633-0b7252ad2405
# ╠═520c2c98-a170-4563-abba-67120d7ccaa3
# ╠═ae703b16-934e-49d9-bdf5-bde468f021be
# ╠═e10f5906-886b-432e-aa87-faddb1290681
# ╠═f36eab1d-075a-452a-9d29-92d8d7856608
# ╠═2b91da86-c959-4fc4-b492-fdd1ef347b27
# ╠═8c7c19d1-80b7-4749-b210-0d347a1a4744
# ╠═380d831e-8237-4bf7-82e6-c94c46ea0765
# ╠═a4569e38-0047-4be2-b53f-bc2558f4996c
# ╠═1e0a2e2f-ec09-4926-af10-57e320b2b213
# ╠═17861ad7-04ee-45c6-a310-51da2009b865
# ╠═9e3332c2-12ae-47df-8b40-61442cb4b479
# ╠═d7ccf03a-835a-472b-914c-5af5f7dc3872
# ╠═77ad7e4e-546b-4402-a39b-78223fd52af9
# ╠═0cd05a44-b82e-4217-a0f4-8b6d3167d936
# ╠═79c03808-bbb3-424f-b678-5858b0fd2d73
# ╠═8e5d760d-4a3c-4326-9e2d-e4b3918bb8c0
# ╠═77b549ac-6204-455d-84b7-5a6653bc89e6
# ╠═658e1360-8604-42f9-9aea-f39c461ed5f5
# ╠═82243cb7-f460-4407-ac55-669402a6db5e
# ╠═ab89293f-725c-4ff9-b683-01a84652e0c7
# ╠═4311478b-0b1e-4939-bf87-63b64e1c70b1
# ╠═04a8aa2e-58e0-46d0-8c61-7fe00b5cc176
# ╠═ce9c3817-ae4c-4419-bda9-aa3b64eba3ce
# ╠═62d5d697-bcbd-4b8a-9d6c-a41907cbfc79
