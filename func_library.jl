using Graphs
import LinearAlgebra
import SparseArrays
using Combinatorics
import COSMO
import CSV
using DataFrames
import MLJ
import Printf
using JuMP
using Convex,Mosek, MosekTools, LinearAlgebra, MathOptInterface, Optim, GLM, Random, Printf



### STUFF FOR CERTIFYING CONVERGENCE

function test_rho(rho,param,env; returnsol = false)
    # function to test if the given convergence rate, rho is feasible for the current algorithm parameters 
    # and the given values of alpha, kappa, and sigma

    # Define optimization variables
    model = Model(Mosek.Optimizer)

    @variable(model, P[1:2,1:2], PSD)
    @variable(model, Q[1:2,1:2], PSD)
    
    kappa,sigma = env

    # Define Algorithm Parameters
     
    alpha, delta, eta, zeta = param

    #alpha = (1-rho)/m

    # Define SS matrices for factorized and switched system

    # Matrices for general 2-state factorization
    A = [1 0; 1 1]
    Ap = [1 0; 0 0]
    B = [-alpha -zeta; 0 -1]
    Bp = [-alpha -zeta; 0 0]
    C = [1 0; delta eta]
    D = [0 -1; 0 0]

    # Define IQCs for gradients and laplacian respectively
    M1 = [-2 kappa+1; kappa+1 -2*kappa]
    M2 = [sigma^2-1 1; 1 -1]

    @variable(model, lam1 >= 0)
    #lam1=1
    @variable(model, lam2 >= 0)
    W1 = kron(M1,[lam1])
    W2 = kron(M2,[lam2])


    # Define LMI quantities
    H1 = [A'*Q*A-rho^2*Q A'*Q*B; B'*Q*A B'*Q*B] + 
        [C[1,:]' D[1,:]'; zeros(1,2) 1 0]'*W1*[C[1,:]' D[1,:]'; zeros(1,2) 1 0] +
        [C[2,:]' D[2,:]'; zeros(1,2) 0 1]'*W2*[C[2,:]' D[2,:]'; zeros(1,2) 0 1]

    H2 = [Ap'*P*Ap-rho^2*P Ap'*P*Bp[:,1]; Bp[:,1]'*P*Ap Bp[:,1]'*P*Bp[:,1]] + 
        [C[1,:]' D[1,1]'; zeros(1,2) 1]'*W1*[C[1,:]' D[1,1]'; zeros(1,2) 1] #+
        #lam2*[C[2,:]' D[2,:]'; zeros(1,2) 0 1]'*M2*[C[2,:]' D[2,:]'; zeros(1,2) 0 1]

    # Set up feasibility problem to satisfy constraints
    #problem = Convex.maximize(eigmin(Q)-eigmax(P))
    @constraint(model, -H1 in PSDCone())
    @constraint(model, -H2 in PSDCone())
    @constraint(model, P-Matrix(I,2,2) in PSDCone())
    @constraint(model, Q-Matrix(I,2,2) in PSDCone())

    # Solve problem and return feasibility status
    optimize!(model)
    #feas = ( termination_status(model) == MOI.OPTIMAL )

    if termination_status(model) == MOI.OPTIMAL
        feas = true
    
    elseif termination_status(model) == MOI.SLOW_PROGRESS
        H1v = value.(H1)
        H2v = value.(H2)
        Pv = value.(P)
        Qv = value.(Q)
        try
            H1_eig = eigmin(-H1v)
            H2_eig = eigmin(-H2v)
            P_eig = eigmin(Pv)
            Q_eig = eigmin(Qv)
            if H1_eig >= 0 && H2_eig >=0 && P_eig > 0 && Q_eig > 0
                feas = true
            else
                #println(H_eig)
                #println(P_eig)
                feas = false
            end
        catch e
            #println("no symmetry")
            feas = false
        end
    
    else
        feas = false
    end


    if returnsol
        soln = Dict()
        if feas
            soln[:H2] = value.(H2)
            soln[:H1] = value.(H1)
            soln[:P] = value.(P)
            soln[:Q] = value.(Q)
            soln[:lam1] = [value(lam1)]
            soln[:lam2] = [value(lam2)]
        end
        return feas, soln
    else
        return feas
    end
end

function test_rho_sync_loss(rho,param,env; returnsol = false)
    # function to test if the given convergence rate, rho is feasible for the current algorithm parameters 
    # and the given values of alpha, kappa, and sigma

    # Define optimization variables
    model = Model(Mosek.Optimizer)

    @variable(model, P[1:3,1:3], PSD)
    @variable(model, Q[1:3,1:3], PSD)
    
    kappa,sigma,p_s = env

    p_l = 1-p_s
   
    # Define Algorithm Parameters
     
    alpha, delta, eta, zeta = param

    #alpha = (1-rho)/m


    # Define SS matrices for factorized and switched system

    # Matrices for general 2-state factorization
    Aps = [1 0 0; 0 0 0; 0 0 0]
    Aqs = [1 0 0; 1 1 0; delta+eta eta 0]
    Bps = [-alpha -zeta; 0 0; 0 0]
    Bqs = [-alpha -zeta; 0 -1; -alpha*delta -(delta*zeta+eta)]

    Apl = [1 0 0; 0 0 0; 0 0 0]
    Aql = [1 0 0; 1 1 0; 0 0 1]
    Bpl = [-alpha -zeta; 0 0; 0 0]
    Bql = [-alpha -zeta; 0 -1; 0 0]

    C = [1 0 0; 0 0 1]
    D = [0 -1; 0 0]

    # Define IQCs for gradients and laplacian respectively
    M1 = [-2 kappa+1; kappa+1 -2*kappa]
    M2 = [sigma^2-1 1; 1 -1]

    @variable(model, lam1 >= 0)
    @variable(model, lam2 >= 0)
    W1 = kron(M1,[lam1])
    W2 = kron(M2,[lam2])


    # Define LMI quantities
    H1 = p_s*[Aqs'*Q*Aqs-rho^2*Q Aqs'*Q*Bqs; Bqs'*Q*Aqs Bqs'*Q*Bqs] + 
        p_l*[Aql'*Q*Aql-rho^2*Q Aql'*Q*Bql; Bql'*Q*Aql Bql'*Q*Bql] +
        [C[1,:]' D[1,:]'; zeros(1,3) 1 0]'*W1*[C[1,:]' D[1,:]'; zeros(1,3) 1 0] +
        [C[2,:]' D[2,:]'; zeros(1,3) 0 1]'*W2*[C[2,:]' D[2,:]'; zeros(1,3) 0 1]

    H2 = p_s*[Aps'*P*Aps-rho^2*P Aps'*P*Bps[:,1]; Bps[:,1]'*P*Aps Bps[:,1]'*P*Bps[:,1]] + 
        p_l*[Apl'*P*Apl-rho^2*P Apl'*P*Bpl[:,1]; Bpl[:,1]'*P*Apl Bpl[:,1]'*P*Bpl[:,1]] + 
        [C[1,:]' D[1,1]'; zeros(1,3) 1]'*W1*[C[1,:]' D[1,1]'; zeros(1,3) 1] #+
        #lam2*[C[2,:]' D[2,:]'; zeros(1,2) 0 1]'*M2*[C[2,:]' D[2,:]'; zeros(1,2) 0 1]

    # Set up feasibility problem to satisfy constraints
    #problem = Convex.maximize(eigmin(Q)-eigmax(P))
    @constraint(model, -H1 in PSDCone())
    @constraint(model, -H2 in PSDCone())
    @constraint(model, P-Matrix(I,3,3) in PSDCone())
    @constraint(model, Q-Matrix(I,3,3) in PSDCone())

    # Solve problem and return feasibility status
    optimize!(model)
    #feas = ( termination_status(model) == MOI.OPTIMAL )
    #println(termination_status(model))

    if termination_status(model) == MOI.OPTIMAL
        feas = true
    
    elseif termination_status(model) == MOI.SLOW_PROGRESS
        H1v = value.(H1)
        H2v = value.(H2)
        Pv = value.(P)
        Qv = value.(Q)
        try
            H1_eig = eigmin(-H1v)
            H2_eig = eigmin(-H2v)
            P_eig = eigmin(Pv)
            Q_eig = eigmin(Qv)
            if H1_eig >= 0 && H2_eig >=0 && P_eig > 0 && Q_eig > 0
                feas = true
            else
                #println(H_eig)
                #println(P_eig)
                feas = false
            end
        catch e
            #println("no symmetry")
            feas = false
        end
    
    else
        feas = false
    end

    if returnsol
        soln = Dict()
        if feas
            soln[:H2] = value.(H2)
            soln[:H1] = value.(H1)
            soln[:P] = value.(P)
            soln[:Q] = value.(Q)
            soln[:lam1] = [value(lam1)]
            soln[:lam2] = [value(lam2)]
        end
        return feas, soln
    else
        return feas
    end
end

function r_up(L,E,m,n,param,edgenum,ind,suc)
    # helper function that produces the input-state matrices row for them memory state r 
    # for a given edge success or loss

    # L, graph Laplacian
    # E, Incidence matrix
    # m, number of edges
    # n, number of nodes
    # param, alg parameters
    # edgenum, current edge number
    # ind, collection of edge indicies in form (i,j)
    # suc, transmission status of packet on current edge

    i,j = ind[edgenum]
    DL = diagm(diag(L))
    IMP = I - 1/n*ones(n,n)
    alpha, delta, eta, zeta = param
    if suc
        A1 = (delta*(1.0*I - zeta*delta*DL) + eta*(IMP*(I-delta*DL)))[j,:]'
        A2 = (delta*(-zeta*eta*DL) + eta*(IMP*(I-eta*DL)))[j,:]'
        Ap = (delta*(-zeta*E) + eta*(-IMP*E))[j,:]'
        B = delta*(-alpha*Matrix(1.0I,n,n))[j,:]'
    else
        A1 = (eta*IMP*(I-delta*DL))[i,:]'
        A2 = -(eta^2*IMP*DL)[i,:]'
        Ap = Matrix(I,m,m)[edgenum,:]' - (eta*IMP*E)[i,:]'
        B = zeros(1,n)
    end 

    return [A1 A2 Ap], B

end

function gen_coll(L,m,n,p_s,param)
    # Generates a collection of state matrices for SH-SVL 
    # with the Laplacian absorbed into the linear dynamics
    # for edgewise loss

    # Keep track which edges exist and compress Laplacian into E 
    # (weighted edge incidence) matrix

    # Takes L, graph Laplacian
    # m, number of edges
    # n, number of nodes
    # p_s, prob of successful packet transmission on each edge
    # param, alg parameters

    ind = []
    E = zeros(n,m)
    offset = 1
    for i = 1:n
        for j = 1:n
            if L[i,j] < 0
                E[i,offset] = L[i,j]
                offset += 1
                push!(ind,(i,j))
            end
        end
    end

    alpha, delta, eta, zeta = param

    #Needed matrices
    DL = diagm(diag(L))
    IMP = I - 1/n*ones(n,n)
    

    #const state matrices
    A11 = (1.0*I - zeta*delta*DL)
    A12 = (-zeta*eta*DL)
    A1p = (-zeta*E)
    B1 = (-alpha*Matrix(1.0I,n,n))
    A21 = (IMP*(I-delta*DL))
    A22 = (IMP*(I-eta*DL))
    A2p = (-IMP*E)

    C = [(I-delta*DL) (-eta*DL) -E]

    Gc = zeros(3*n+m,3*n+m)
    Gc[1:3*n+m,1:3*n+m] = zeros(3*n+m,3*n+m)
    Gc[1:n,1:2n+m] = [A11 A12 A1p]
    Gc[n+1:2n,1:2n+m] = [A21 A22 A2p]

    Gc[1:n,2n+m+1:end] = B1
    Gc[2n+m+1:end,1:2n+m] = C

    # get all 2^m combinations of edgewise packet loss
    iter = combinations(ind)

    # create a collection of probabilities and state matrices for 
    # all possible combos of edgewise packet loss
    coll = []
    for succs in enumerate(iter)
        Gi = copy(Gc)
        for i = 1:m
            currind = ind[i]
            if currind in succs[2]
                Ap, Bp = r_up(L,E,m,n,param,i,ind,true)
            else
                Ap, Bp = r_up(L,E,m,n,param,i,ind,false)
            end
            Gi[2n+i,:] = [Ap Bp]
        end
        ns = length(succs[2])
        p = p_s^ns*(1-p_s)^(m-ns)
        push!(coll,(p,Gi))
    end
    # include the case where all packets are lost
    Gi = copy(Gc)
    for i = 1:m
        Ap, Bp = r_up(L,E,m,n,param,i,ind,false)
        Gi[2n+i,:] = [Ap Bp]
    end
    p = (1-p_s)^(m)
    push!(coll,(p,Gi))

    return coll

end

function gen_coll_sync_loss(L,n,p_s,param)
    # Generates a collection of state matrices for SH-SVL 
    # with the Laplacian absorbed into the linear dynamics
    # for synchronous loss

    # Takes L, graph Laplacian
    # n, number of nodes
    # p_s, prob of successful packet transmission in network
    # param, alg parameters

    alpha, delta, eta, zeta = param
    #Needed matrices
    IMP = I - 1/n*ones(n,n)
    eye = Matrix(I,n,n)

    As = [eye 0*eye -zeta*L;
        IMP IMP -L;
        delta*I+eta*IMP eta*IMP -(delta*zeta+eta)L]
    Bs = [-alpha*eye; 0*eye; -alpha*delta*eye]

    Al = [eye eye*0 -zeta*L;
        IMP IMP -L;
        0*eye 0*eye eye]
    Bl = [-alpha*eye; 0*eye; 0*eye]

    C = [eye 0*eye -L]


    Gs = [As Bs; C 0*eye]
    Gl = [Al Bl; C 0*eye]

    coll = [(p_s,Gs) (1-p_s,Gl)]
    return coll

end

function test_rho_edgewise_loss(rho,coll,env; returnsol = false)
    # For pairwise losses
    # function to test if the given convergence rate, rho is feasible for the current algorithm parameters 
    # and the given values of alpha, kappa, and sigma

    # Define optimization variables
    model = Model(Mosek.Optimizer)

    kappa,n,tol = env
    s = size(coll[1][2])[2]
    n = Int(n)
    m = s-3*n
    @variable(model, P[1:2n+m,1:2n+m], PSD)
    
    


    # Define SS matrices for factorized and switched system

    # Matrices for general 2-state factorization


    C = coll[1][2][2n+m+1:end,1:2n+m]
    D = coll[1][2][2n+m+1:end,2n+m+1:end]


    # Define IQCs for gradients and laplacian respectively
    M1 = [-2 kappa+1; kappa+1 -2*kappa]

    @variable(model, lam1 >= 0)
    W1 = kron(M1,lam1*Matrix(I,n,n))
  


    # Define LMI quantities
    H = [C D;zeros(n,2n+m) Matrix(I,n,n)]'*W1*[C D;zeros(n,2n+m) Matrix(I,n,n)]

    for i = 1:length(coll)
        pi,Gi = coll[i]
        Ai = Gi[1:2n+m,1:2n+m]
        Bi = Gi[1:2n+m,2n+m+1:end]
        H += pi*[Ai'*P*Ai-rho^2*P Ai'*P*Bi; Bi'*P*Ai Bi'*P*Bi]
    end

    # Set up feasibility problem to satisfy constraints
    #problem = Convex.maximize(eigmin(Q)-eigmax(P))
    @constraint(model, -H in PSDCone())
    @constraint(model, P-Matrix(I,2n+m,2n+m) in PSDCone())
    #@constraint(model, P in PSDCone())


    # Solve problem and return feasibility status
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        feas = true
    
    elseif termination_status(model) == MOI.SLOW_PROGRESS
        Hv = value.(H)
        Pv = value.(P)
        try
            H_eig = eigmin(-Hv)
            P_eig = eigmin(Pv)
            if H_eig >= 0 && P_eig > 0
                feas = true
            else
                #println(H_eig)
                #println(P_eig)
                feas = false
            end
        catch e
            #println("no symmetry")
            feas = false
        end
    
    else
        feas = false
    end
    #println(rho)
    #println(termination_status(model))
    if returnsol
        soln = Dict()
        if feas
            soln[:H] = value.(H)
            soln[:P] = value.(P)
            soln[:lam1] = [value(lam1)]
        end
        return feas, soln
    else
        return feas
    end
end


### STUFF FOR BISECTION ON RATE

function find_rho(f,param,env, tol=1e-5)
    # bisection search for min rho that is feasible up to given tol

    # f, SDP solver test function (e.g., test_rho_jump_nat)
    # param, alg parameters
    # env, alg environment (e.g., kappa and sigma)

    a,b = 0,1
    if f(a,param,env)[1]
        return a
    end
    if !f(b,param,env)[1]
        return b
    end
    while (b-a) > tol
        c = (a+b)/2
        if f(c,param,env)[1]
            b = c
        else
            a = c
        end
    end
    return b
end

function find_rho_var(f,coll_var,param,env, tol=1e-5)
    # bisection search for min rho that is feasible up to given tol
    # for use with gen_coll

    kappa,n,m,p_s,tol2 = env
    L = coll_var
    n = Int(n)
    m = Int(m)
    env = [kappa,n,tol2]
    coll = gen_coll(L,m,n,p_s,param)
    a,b = 0,1
    if f(a,coll,env)[1]
        return a
    end
    if !f(b,coll,env)[1]
        return b
    end
    while (b-a) > tol
        c = (a+b)/2
        if f(c,coll,env)[1]
            b = c
        else
            a = c
        end
    end
    return b
end

function find_rho_sync_L(f,L,param,env; tol=1e-5)
    # bisection search for min rho that is feasible up to given tol
    # for use with gen_coll_sync_loss

    kappa,n,p_s,tol2 = env
    n = Int(n)
    env = [kappa,n,tol2]
    coll = gen_coll_sync_loss(L,n,p_s,param)
    a,b = 0,1
    if f(a,coll,env)[1]
        return a
    end
    if !f(b,coll,env)[1]
        return b
    end
    while (b-a) > tol
        c = (a+b)/2
        if f(c,coll,env)[1]
            b = c
        else
            a = c
        end
    end
    return b
end



### FUNCTIONS FOR SOLVING SVL

function SVL_parameters(m,L,σ,ε)
    # Algorithm 2 in Bryan's 2020 TCNS paper
    # (here L is not the Laplacian but rather the Lipschitz constant)
    #
    # Also, must start with ρ₁=(L-m)/(L+m) and not ρ₁=0
    # because otherwise LHS of (12b) can be negative?
    κ = L/m
    ρ₁ = (L-m)/(L+m)
    ρ₂ = 1.0
    ρ = 0.5  # must define before while block
    β = 1.0  # must define before while block
    while ρ₂-ρ₁ > ε
        ρ = (ρ₁+ρ₂)/2
        η = 1 + ρ - κ*(1-ρ)
        s₀ = η*(1-ρ^2)^2*(η-(3-η)*η*ρ+2*(1-η)*ρ^2+2ρ^3)
        s₁ = -(1-ρ^2)*(η^3*ρ+4ρ^5-2η*ρ^2*(2ρ^2+ρ-3)+η^2*(4ρ^3-4ρ^2-6ρ+3))
        s₂ = 3η*(1-ρ)^2*(1+ρ)*(2ρ^2+η)
        s₃ = (2ρ^2+η)*(2ρ^3-η)
        β = real_cubic_roots(s₃,s₂,s₁,s₀)
        if length(β) > 1
            # find out which one satisfies (12a)
            β = β[findfirst(x->(2x-(1-ρ)*(κ+1))*(x-1+ρ^2)<0,β)]
        end
        σhat = ρ^2*((β-1+ρ^2)/(β-1+ρ))*((2-η-2β)/(2ρ^2*β-(1-ρ^2)*η))
        σhat *= ((2ρ^2+η)*β-(1-ρ^2)*η)/((1+ρ)*(η-2η*ρ+2ρ^2)-(2ρ^2+η)*β)
        σhat = √σhat
        if σhat < σ
            ρ₁ = ρ
        else
            ρ₂ = ρ
        end
    end
    α = (1-ρ)/m
    γ = β + 1
    δ = 1.0
    return α, β, γ, δ, ρ
end


function real_cubic_roots(a,b,c,d)
    # compute real roots of cubic ax^3+bx^2+cx+d
    # (a is nonzero, real coefficients)
    iszero(a) && @warn "Leading coefficient of cubic is zero"
    p = (3a*c-b^2)/(3a^2)
    q = (2b^3-9a*b*c+27a^2*d)/(27a^3)
    Δ = -(4p^3+27q^2)
    if Δ < 0
        # only one real root
        if p < 0
            t = -2*sign(q)*√(-p/3)*cosh(acosh((-1.5*abs(q)/p)*√(-3/p))/3)
        elseif p > 0
            t = -2*√(p/3)*sinh(asinh((1.5*q/p)*√(3/p))/3)
        else
            t = -sign(q)*(abs(q))^(1/3)
        end
        return t - b/(3a)
    end
    # three real roots
    if iszero(p)
        t = [0.0,0.0,0.0]
    else
        t = 2*√(-p/3)*cos.(acos((1.5*q/p)*√(-3/p))/3 .- 2*π*[0,1,2]/3)
    end
    return t .- b/(3a)
end


### SIMULATION OF UNSTABLE DISTRIBUTED OPTIMIZATION

function drop_packets!(recv,drop_prob,recv_prob)
    # Markovian packet loss model:
    # drop_prob = prob of dropped packet given prev packet was dropped
    # recv_prob = prob of received packet given prev packet was received
    #
    # inputs are matrices, so different edges can have different probs
    # function modifies recv input matrix, diagonal set to true
    R = rand(Float64, size(recv))
    @. recv = R < (recv*recv_prob + !recv*(1-drop_prob))
    recv[diagind(recv)] .= true
    return nothing
end


function unstable_gd(G,F,d,L,max_iter;random_init=false, drop_prob=zeros(size(L)),
    recv_prob=1.0.-drop_prob,predict=0, xopt::Array, tolerance::Float64 = 1e-5)
    # Takes in state matrices G
    # Stacked function gradients F (defined as a function)
    # dimension of decsion variable d
    # Graph Laplacian L
    # Number of iterations iter

    # if predict is non-zero it should be eta (the correction term coefficient)
    # Assumes Dy is nonzero and correction term is of the form of SH-SVL

    n = size(L,1)
    eye = Matrix(I,n,n)

    # Projected System
    #P = 1/n*ones(n,n)
    #Q = [eye zeros(n,n); zeros(n,n) eye-P]
    #A = Q*kron(G[1:2,1:2],eye)
    #B = Q*kron(G[1:2,3:4],eye)

    A = kron(G[1:2,1:2],eye)
    B = kron(G[1:2,3:4],eye)
    Cx = kron(G[3,1:2],eye)
    Dx = kron(G[3,3:4],eye)
    Cy = kron(G[4,1:2],eye)

    w = zeros(2n,d,max_iter+1)
    x = zeros(n,d,max_iter)
    y = zeros(n,d,max_iter)
    u = zeros(n,d,max_iter)
    v = zeros(n,d,max_iter)
    h = zeros(n,d,max_iter)

    time = max_iter

    # init states
    if random_init
        w[:,:,1] = rand(2n,d)
    else
        w[:,:,1] = zeros(2n,d)#rand(2n,d)
    end

    # do we ever drop packets?
    drop = any((drop_prob .> 0) .| (recv_prob .< 1))
    
    Recv = rand(Bool,n,n)
    if drop
        println("Entering Lossy Channel")
        # init edge states
        M = rand(n,d,n)
        for k = 1:max_iter
            y[:,:,k] = Cy'w[:,:,k]

            # compute v given packet drops
            drop_packets!(Recv,drop_prob,recv_prob)
            for i = 1:n
                R = @view Recv[:,i]
                if k >= 2
                    M[:,:,i] = R.*y[:,:,k] + (.!R).*(predict*x[i,:,k-1]'.+M[:,:,i])
                    v[i,:,k] = L[i,:]'M[:,:,i]
                    h[i,:,k] = v[i,:,k]' - L[i,:]'y[:,:,k]
                else
                    M[:,:,i] = R.*y[:,:,k] + (.!R).*M[:,:,i]
                    v[i,:,k] =  L[i,:]'M[:,:,i]
                    h[i,:,k] = v[i,:,k]' - (L[i,:]'*y[:,:,k])
                end
            end
            x[:,:,k] = Cx'w[:,:,k] + Dx'*[u[:,:,k];v[:,:,k]]
            u[:,:,k] = F(x[:,:,k])
            w[:,:,k+1] = A*w[:,:,k] + B*[u[:,:,k];v[:,:,k]]

            e = [maximum([norm(x[i,:,k].-xopt,Inf) for i=1:n]) for k=k]
            if e[1] < tolerance
                time = k
                break
        end

        end
    else
        for k = 1:max_iter
            y[:,:,k] = Cy'w[:,:,k]
            v[:,:,k] = L*y[:,:,k]
            x[:,:,k] = Cx'w[:,:,k] + Dx'*[u[:,:,k];v[:,:,k]]
            u[:,:,k] = F(x[:,:,k])
            w[:,:,k+1] = A*w[:,:,k] + B*[u[:,:,k];v[:,:,k]]

            e = [maximum([norm(x[i,:,k].-xopt,Inf) for i=1:n]) for k=k]
            if e[1] < tolerance
                time = k
                break
            end
        end
    end
    if drop
        return u,v,w,x,y,h,time
    else
        return u,v,w,x,y,time
    end
end


### DISTRIBUTED LOGISTIC REGRESSION EXAMPLE

abstract type AbstractModelData{T} end

# Defines terms
struct LogisticRegression{T<:Real} <: AbstractModelData{T}
    # f = Σ ln(1+exp(-γh⋅x)) + μ₁|x|₁ + 0.5μ₂|x|₂² (elastic net regularization)
    # x ∈ R^p
    # sum is over s samples
    # H is an p-by-s matrix
    # Γ is a vector of s labels in {-1,+1}
    # μ₁ is the ℓ₁ regularization parameter
    # μ₂ is the ℓ₂ regularization parameter
    # m, L are the convexity parameters
    H::Matrix{T}
    Γ::Vector{T}
    μ₁::T
    μ₂::T
    p::Int
    s::Int
    m::T
    L::T
    LogisticRegression{T}(H,Γ,μ₁,μ₂) where {T<:Real} =
        new{T}(H,Γ,μ₁,μ₂,size(H,1),length(Γ),μ₂,μ₂+0.25*opnorm(H)^2)
end

function logistic_regression_COSMO_chip_data(n,d=6,μ₁=0.1,μ₂=0.1)
    # random partition samples into n groups
    # d = polynomial degree for mapping 2d features (d=6 in COSMO docs)
    # μ₁ is the ℓ₁ regularization parameter, μ₁|x|₁
    # μ₂ is the ℓ₂ regularization parameter, 0.5μ₂|x|₂²
    # (Note: we are using the standard ℓ₂ regularization, whereas the
    # example in the COSMO docs regularizes using the ℓ₂-norm rather
    # than the square of the norm. So to get a solution close to the one
    # in the COSMO docs we take μ₂=0.1 instead of μ₂=1.)
    df = DataFrame(CSV.File("data/chip_data.txt",header=false)) #|> DataFrame
    x = Matrix(df[:,1:2])
    y = df[:,3]
    S = length(y)
    s = rand_partition(S,n)
    p = binomial(d+2,d)  # number of features for 2d-data
    H = [zeros(p,s[i]) for i=1:n]
    X = zeros(p,S)
    Γ = [ones(s[i]) for i=1:n]
    ℓ = 0
    for i = 1:n, j = 1:s[i]
        ℓ += 1
        X[:,ℓ] .= poly_map_features(x[ℓ,:],d)
        H[i][:,j] .= X[:,ℓ]
        iszero(y[ℓ]) && (Γ[i][j] = -1)
    end
    y[iszero.(y)].=-1
    data = [LogisticRegression{Float64}(H[i],Γ[i],μ₁/n,μ₂/n) for i=1:n]
    return data, X, y, H, Γ
end

function poly_map_features(x,d)
    # general form of chip_map_features
    return [prod(x.^a) for i=0:d for a in multiexponents(length(x),i)]
end

function rand_partition(S,n)
    # random partition of S into n parts, with each part
    # containing at least one entry
    s = ones(Int,n)

    # seed here
    Random.seed!(13)
    r = rand(1:n,S-n)
    for i = 1:S-n
        s[r[i]] += 1
    end
    return s
end

function LogReg(x,H,G)
    # computes the cost for logistic regression
    n,d = size(x)
    u = zeros(n,d)
    for i = 1:n
        h = H[i]
        g = G[i]
        u[i,:] = 2/n*x[i,:] + sum(-g[j]*exp(-g[j]*x[i,:]'h[:,j])*h[:,j]/(1+exp(-g[j]*x[i,:]'h[:,j])) 
                                for j=1:length(g))
    end
    return u
end

function estL(h,n)
    # Estimates the Lipschitz constant for the logistic regression problem
    p,l = size(h)
    opnorm((2/n)*I + 0.25*sum(h[:,j]*h[:,j]' for j=1:l))
end



### GRAPH STUFF


function spectral_gap(L)
    # J = I - 11'/n
    # find λ to minimize σ = ‖J-λL‖
    # returns λ, σ
    n = size(L,1)
    J = I - ones(n,n)/n
    model = COSMO.Model()
    A = [vec(Matrix(1.0I,2n,2n)) vec([0I -L';-L 0I])]
    b = vec([0I J;J 0I])
    constraint = COSMO.Constraint(A,b,COSMO.PsdCone)
    settings = COSMO.Settings(eps_abs=1e-5,eps_rel=1e-5,verbose=true)
    COSMO.assemble!(model,zeros(2,2),[1.0,0.0],constraint,settings=settings)
    results = COSMO.optimize!(model)
    results.status == :Solved || @warn "COSMO did not return :Solved"
    return results.x[2], results.x[1]
end

function my_graph()
    g = SimpleDiGraph(8)
    add_edge!(g,1,2)
    add_edge!(g,2,4)
    add_edge!(g,2,5)
    add_edge!(g,2,6)
    add_edge!(g,2,8)
    add_edge!(g,3,1)
    add_edge!(g,3,2)
    add_edge!(g,3,5)
    add_edge!(g,3,7)
    add_edge!(g,4,3)
    add_edge!(g,4,6)
    add_edge!(g,5,6)
    add_edge!(g,6,3)
    add_edge!(g,6,4)
    add_edge!(g,6,5)
    add_edge!(g,7,3)
    add_edge!(g,7,4)
    add_edge!(g,7,6)
    add_edge!(g,7,8)
    add_edge!(g,8,2)
    add_edge!(g,8,3)
    return g
end

# sum across row get zero
function equal_in_laplacian(g::SimpleGraph)
    # in-Laplacian with equal node in-weights that sum to one
    A = adjacency_matrix(g)
    A *= Diagonal(1 ./ indegree(g))
    return Diagonal(vec(sum(A,dims=1))) - A
end

function equal_out_laplacian(g::SimpleGraph)
    # out-Laplacian with equal node out-weights that sum to one
    A = adjacency_matrix(g)
    A = A ./ outdegree(g)
    return Matrix(Diagonal(vec(sum(A,dims=2))) - A)
end

function scaled_laplacian(g::SimpleGraph, s)
    A = adjacency_matrix(g)
    n,n = size(A)
    return s.*(Diagonal(A*ones(n))-A)
end

function basic_sigma(L)
    (n,n) = size(L)
    return opnorm(I-1/n*ones(n,n)-L)
end

function geo_graph(n,r)
    V = rand(n,2)
    A = [norm(i-j) for i in eachrow(V), j in eachrow(V)].<r
    A = A-I
    dout = vec(sum(A,dims=2))
    L = Diagonal(dout)-A
    L = L./dout
end

function calc_ratio(opt_rho, lossless_rho, lossy_rho)
    lossless_ratio = log.(lossless_rho)./log.(opt_rho)
    lossy_ratio = log.(lossy_rho)./log.(opt_rho)

    return lossless_ratio, lossy_ratio
end

function window_rho(e,e2,t1,t2)
    mid1 = Int64(round(t1/3))
    mid2 = Int64(round(t2/3))
    
    interval1 = Int64(round(t1/5))
    interval2 = Int64(round(t2/5))

    time1 = 1:1:t1
    time2 = 1:1:t2

    correlation = 0

    rho1 = 0
    rho2 = 0

    data1 = DataFrame()
    data2 = DataFrame()

    while correlation < 0.99 && mid1 < (t1 - (interval1/2))
        
        data1 = DataFrame(X=time1[mid1-Int64(round(interval1/2)):mid1+Int64(round(interval1/2))],Y=log.(e[mid1-Int64(round(interval1/2)):mid1+Int64(round(interval1/2))]))
        ols = lm(@formula(Y~X),data1)
        rho1 = abs(coef(ols)[2])
        correlation = GLM.r2(ols)
        mid1 = Int64(round(mid1 + interval1/2))
    end

    correlation = 0
    while correlation < 0.99 && mid2 < (t2 - (interval2/2))
        
        data2 = DataFrame(X=time2[mid2-Int64(round(interval2/2)):mid2+Int64(round(interval2/2))],Y=log.(e2[mid2-Int64(round(interval2/2)):mid2+Int64(round(interval2/2))]))
        ols = lm(@formula(Y~X),data2)
        rho2 = abs(coef(ols)[2])
        correlation = GLM.r2(ols)

        mid2 = Int64(round(mid2 + interval2/2))
    end

    rho1 = rho1^(1/nrow(data1))
    rho2 = rho2^(1/nrow(data2))

    return rho1, rho2
end

function optimize_parameters(sigma::Float64, kappa::Float64, warmstart::Bool, prev_optparam)
    if warmstart ==true
        localenv = [kappa, sigma]#, 1-pl] kappa = L/m
        tol = 1e-5

        global k = 0
        global param_init = prev_optparam # warmstarting will speed the following code up
        max_iter = 40000
        while k < max_iter && find_rho(test_rho,param_init,localenv)==1.0
            global param_init = Diagonal([1,2,2,2])*rand(4)
            global k += 1
        end
    else
        localenv = [kappa, sigma]#, 1-pl] kappa = L/m
        tol = 1e-5

        global k = 0
        global param_init = Diagonal([1,2,2,2])*rand(4) # warmstarting will speed the following code up
        max_iter = 40000
        while k < max_iter && find_rho(test_rho,param_init,localenv)==1.0
            global param_init = Diagonal([1,2,2,2])*rand(4)
            global k += 1
        end
    end

    if k == max_iter
        rho = 1
        optparam = copy(param_init)
        @printf("max iter reached\n")
    else
        res = Optim.optimize(param_val -> find_rho(test_rho, param_val, localenv), param_init, Optim.NelderMead())
        rho = Optim.minimum(res)
        optparam = Optim.minimizer(res)
    end
    @printf("rho = %0.4f\n", rho)
    println("Alg 2 opt params")
    println(optparam)

    return optparam, rho
end

function theoretical_lower_bound(kappa, sigma)
    return maximum([(kappa.-1)./(kappa.+1) sigma], dims=2)
end

function remove_duplicates(path,duplicate_var)

    df = DataFrame(CSV.File(path, header=true))
    numrows = size(df,1)
    
    i = 1
    while i < numrows
        if df[i,duplicate_var] == df[i+1,duplicate_var]
            if df[i,"sigma_error"] < df[i+1, "sigma_error"]
                delete!(df,i+1)
            else
                delete!(df,i)
            end
            numrows-=1
        else
            i += 1
        end
    end

    CSV.write(path, df)

end
