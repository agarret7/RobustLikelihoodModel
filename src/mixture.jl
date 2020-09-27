import Gen

struct Mixture{T} <: Gen.Distribution{T}
    components::Vector{Gen.Distribution{T}}
    #has_output_grad::Bool # only if all of them do
end

#function Mixture{T}(components...)
    #return Mixture{T}(components, all(map(has_output_grad, components)))
#end

function Gen.logpdf(
        dist::Mixture{T}, x::T, weights::Vector{Float64},
        arg_tuples::Vector) where {T}
    ls = Vector{Float64}(undef, length(dist.components))
    for i=1:length(dist.components)
        ls[i] = Gen.logpdf(dist.components[i], x, arg_tuples[i]...) + log(weights[i])
    end
    Gen.logsumexp(ls)
end

function Gen.random(
        dist::Mixture, weights::Vector{Float64},
        arg_tuples::Vector)
    i = Gen.categorical(weights)
    Gen.random(dist.components[i], arg_tuples[i]...)
end

# TODO to support derivatives with respect to arguments,
# we need to support tuple-valued derivatives 

function Gen.logpdf_grad(
        dist::Mixture{T}, x::T, weights::Vector{Float64},
        arg_tuples::Vector) where {T}
    # TODO support gradients with respect to weights only for now
    error("not implemented")
end

(dist::Mixture)(weights, arg_tuples) = Gen.random(dist, weights, arg_tuples)
Gen.is_discrete(dist::Mixture) = Gen.is_discrete(dist.components[1])
Gen.has_output_grad(dist::Mixture) = false
Gen.has_argument_grads(dist::Mixture) = (true, false)
