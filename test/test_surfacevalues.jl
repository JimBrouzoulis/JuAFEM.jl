function test_surfacevectorvalues()
    x = [
        Vec{2,Float64}((0.0, 0.0)),
        Vec{2,Float64}((6.0, 0.0)),
        Vec{2,Float64}((6.0, 1.0)),
        Vec{2,Float64}((0.0, 1.0)),
        ]
    func_interpol = CohesiveZone{1,RefCube,1,2}()
    quad_rule = QuadratureRule{1,RefCube}(:lobatto,2)
    cv = SurfaceVectorValues(quad_rule, func_interpol)
    ndim = JuAFEM.getdim(func_interpol)
    n_basefuncs = getnbasefunctions(func_interpol)

    # fe_valtype == CellScalarValues && @test getnbasefunctions(cv) == n_basefuncs
    @test getnbasefunctions(cv) == n_basefuncs * JuAFEM.getspacedim(func_interpol)

    # x, n = valid_coordinates_and_normals(func_interpol)
    reinit!(cv, x)

    # Test computation of the jump vector
    u_vector = [2., 0., 3., 0.,
                4., 5., 4., 3.]

    val_qp1 = function_value(cv, 1, u_vector)
    @test val_qp1[1] ≈ 2.0
    @test val_qp1[2] ≈ 3.0
    val_qp2 = function_value(cv, 2, u_vector)
    @test val_qp2[1] ≈ 1.0
    @test val_qp2[2] ≈ 5.0

    # test integration
    area = 0.0
    for i in 1:getnquadpoints(cv)
        area += JuAFEM.getdetJdA(cv,i)
    end
    @test area ≈ 6.0

    @test spatial_coordinate(cv, 1, x) ≈ Vec{2}((0., 0.5))
    @test spatial_coordinate(cv, 2, x) ≈ Vec{2}((6., 0.5))

end

@testset "SurfaceValues" begin
    test_surfacevectorvalues()
end
