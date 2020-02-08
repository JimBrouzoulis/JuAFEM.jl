
abstract type SurfaceInterpolation{dim,shape,order,dim_s} <:  Interpolation{dim,shape,order} end

struct CohesiveZone{dim,shape,order,dim_s} <: SurfaceInterpolation{dim,shape,order,dim_s} end

JuAFEM.getnbasefunctions(::CohesiveZone{dim,shape,1,dim_s}) where {dim,shape,order,dim_s} = 4
JuAFEM.nvertexdofs(::CohesiveZone) = 1

"""
Return the refeerence coordinates for the fictious mid-surface
"""
JuAFEM.reference_coordinates(::CohesiveZone{1,shape,order,dim_s})  where {dim,shape,order,dim_s} = reference_coordinates(Lagrange{1,RefCube,order}())

"""
Return the spatial dimension of a `SurfInterpolation`
"""
@inline getspacedim(ip::SurfaceInterpolation{dim,shape,order,dim_s}) where {dim,shape,order,dim_s} = dim_s


function value(ip::CohesiveZone{1,RefCube,1,dim_s}, i::Int, ξ::Vec{1}) where {dim_s}
    """
    Shape function values are defined such that ∑ Nᵢ(ξ) * aᵢ = j(ξ) (spatial jump)
    Note: current node numbering is:
    4__________3
    |          |
    |__________|
    1          2
    """
    ξ_x = ξ[1]
    i == 1 && return -(1 - ξ_x) * 0.5
    i == 2 && return -(1 + ξ_x) * 0.5
    i == 3 && return (1 + ξ_x) * 0.5
    i == 4 && return (1 - ξ_x) * 0.5
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

function mid_surf_value(ip::CohesiveZone{1,RefCube,1,dim_s}, i::Int, ξ::Vec{1}) where {dim_s}
    """
    Shape function values are defined such that ∑ Nᵢ(ξ) * xᵢ = x̄(ξ) (mid-surface)
    Note: current node numbering is:
    4__________3
    |          |
    |__________|
    1          2
    """
    ξ_x = ξ[1]
    i == 1 && return (1 - ξ_x) * 0.25
    i == 2 && return (1 + ξ_x) * 0.25
    i == 3 && return (1 + ξ_x) * 0.25
    i == 4 && return (1 - ξ_x) * 0.25
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
