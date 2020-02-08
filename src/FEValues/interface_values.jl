



struct SurfaceVectorValues{dim,dim_s,T<:Real,refshape<:JuAFEM.AbstractRefShape} <: CellValues{dim,T,refshape}
    N::Matrix{Tensor{dim,dim_s,T}}
    dNdξ::Matrix{Tensor{dim,dim_s,T}}
    detJdA::Vector{T}
    M::Matrix{T}  # Shape values for geometric interp
    dMdξ::Matrix{Vec{dim,T}}
    qr_weights::Vector{T}
    covar_base::Vector{Tensor{2,dim_s,T}}
end


function SurfaceVectorValues(quad_rule::QuadratureRule, func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)
    SurfaceVectorValues(Float64, quad_rule, func_interpol, geom_interpol)
end


function SurfaceVectorValues(::Type{T},
    quad_rule::QuadratureRule{dim,shape},
    func_interpol::SurfaceInterpolation{dim,shape,order,dim_s},
    geom_interpol::SurfaceInterpolation{dim,shape,order,dim_s}=func_interpol) where {dim,dim_s,T,shape<:JuAFEM.AbstractRefShape,order}

    @assert JuAFEM.getdim(func_interpol) == JuAFEM.getdim(geom_interpol)
    @assert JuAFEM.getrefshape(func_interpol) == JuAFEM.getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim_s
    N    = fill(zero(Tensor{dim,dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Tensor{dim,dim_s,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    covar_base = fill(zero(Tensor{2,dim_s,T}) * T(NaN), n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp, N_temp = JuAFEM.gradient(ξ -> JuAFEM.value(func_interpol, basefunc, ξ), ξ, :all)
            for comp in 1:dim_s
                N_comp = zeros(T, dim_s)
                N_comp[comp] = N_temp
                N[basefunc_count, qp] = Vec{dim_s,T}((N_comp...,))

                dN_comp = zeros(T, dim_s, dim)
                dN_comp[comp, :] = dNdξ_temp
                dNdξ[basefunc_count, qp] = Tensor{1,dim_s,T}((dN_comp...,))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = JuAFEM.gradient(ξ -> JuAFEM.mid_surf_value(geom_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdA = fill(T(NaN), n_qpoints)
    SurfaceVectorValues{dim,dim_s,T,shape}(N, dNdξ, detJdA, M, dMdξ, quad_rule.weights, covar_base)
end

getn_scalarbasefunctions(cv::SurfaceVectorValues{dim,dim_s}) where {dim,dim_s} = size(cv.N, 1) ÷ dim_s

@inline getdetJdA(cv::SurfaceVectorValues, q_point::Int) = cv.detJdA[q_point]

function spatial_coordinate(fe_v::SurfaceVectorValues{dim,dim_s}, q_point::Int, x::AbstractVector{Vec{dim_s,T}}) where {dim,T, dim_s}
    n_base_funcs = JuAFEM.getngeobasefunctions(fe_v)
    @assert length(x) == n_base_funcs
    vec = zero(Vec{dim_s,T})
    @inbounds for i in 1:n_base_funcs
        vec += geometric_value(fe_v, q_point, i) * x[i]
    end
    return vec
end



function reinit!(cv::SurfaceVectorValues{dim,dim_s}, x::AbstractVector{Vec{dim_s,T}}) where {dim,dim_s,T}
    n_geom_basefuncs = JuAFEM.getngeobasefunctions(cv)
    n_func_basefuncs = JuAFEM.getn_scalarbasefunctions(cv)
    @assert length(x) == n_geom_basefuncs
    isa(cv, CellVectorValues) && (n_func_basefuncs *= dim)


    @inbounds for i in 1:length(cv.qr_weights)
        w = cv.qr_weights[i]
        fecv_J = zero(Tensor{dim,dim_s})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] * cv.dMdξ[j, i].data[1]
        end
        detJ = norm(fecv_J)
        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        G = zeros(T,dim_s,dim_s)
        G[:,1] = fecv_J
        G[:,2] = [-fecv_J[2], fecv_J[1]]
        cv.covar_base[i] = Tensor{2,dim_s,T}(G)
        detJ = sqrt(det(cv.covar_base[i]))
        cv.detJdA[i] = detJ * w

    end
end


function function_value(fe_v::SurfaceVectorValues{dim,dim_s}, q_point::Int, u::AbstractVector{T}, dof_range::UnitRange = 1:length(u)) where {dim,T,dim_s}
    n_base_funcs = JuAFEM.getn_scalarbasefunctions(fe_v)
    n_base_funcs *= dim_s
    @assert length(dof_range) == n_base_funcs
    @boundscheck checkbounds(u, dof_range)
    val = zero(Vec{dim_s,T})
    @inbounds for (i, j) in enumerate(dof_range)
        val += shape_value(fe_v, q_point, i) * u[j]
    end
    return val
end
