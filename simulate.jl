using LinearAlgebra, Plots, Convex, Mosek, MosekTools, JuMP, LaTeXStrings, Printf, Measures, Serialization, Graphs, CurveFit

include("func_library.jl")

function simulate_file(filename::String, opt_problem::Function, problem_type::String, iter::Int64, optimize_params::Bool, warmstart::Bool, 
    prev_optparam::Array,pl::Float64,alpha::Float64=0.1)

    varying_sigma = false
    if cmp(split(filename,"-")[1], "s") == 0
        varying_sigma = true
    end

    metadata = DataFrame(CSV.File("data/"*filename*".csv",header=true))
    graphs_filename = "data/"*filename*".lg"
    
    metadata.opt_rho = missings(Float64, nrow(metadata))
    metadata.lossless_rho = missings(Float64, nrow(metadata))
    metadata.lossy_rho = missings(Float64, nrow(metadata))

    metadata.lossless_t = missings(Int64, nrow(metadata))
    metadata.lossy_t = missings(Int64, nrow(metadata))
    metadata.lossless_e = missings(Float64, nrow(metadata))
    metadata.lossy_e = missings(Float64,nrow(metadata))
    metadata.kappa = missings(Float64,nrow(metadata))

    prev_optparam = Diagonal([1,2,2,2])*rand(4)

    mkpath("figures/"*filename)

    sigma_sum = 0
    max_kappa = 0

    for j in 1:size(metadata,1)
        name = String(metadata[j,"name"])
        graph = Graphs.loadgraph(graphs_filename,name)
        sigma = metadata[j,"sigma"]

        sigma_sum += sigma

        A = adjacency_matrix(graph)
        n,n = size(A)

        data,X,Y,H,Gam = opt_problem(n,6,0,1)
        X = X'
        d = size(X)[2]
        
        if problem_type == "LogisticRegression"
            f = x -> LogReg(x,H,Gam) # Calculate the cost
            m = 2/n
            Lip = maximum([estL(H[i],n) for i = 1:n])
            kappa = Lip/m
            @printf("L = %0.4f, and kappa = %0.4f\n", Lip, kappa)
        end

        if kappa > max_kappa
            max_kappa = kappa
        end

        metadata[j,"kappa"] = kappa
    end

    if varying_sigma == false
        sigma = sigma_sum/size(metadata,1)
        kappa = max_kappa
    
        if optimize_params == true
            optparam, rho = optimize_parameters(sigma,kappa,false,[])
            alpha, delta, eta, zeta = optparam
        else
            # Default NIDS parameters
            delta = 0.5
            eta = 0.5
            zeta = 1.0
        
        end
    end

    for i in 1:size(metadata,1)
        name = String(metadata[i,"name"])
        graph = Graphs.loadgraph(graphs_filename,name)
        sigma = metadata[i,"sigma"]
        scale = metadata[i, "scale"]
        kappa = metadata[i, "kappa"]
        
        @printf("sigma = %0.5f\n", sigma)
        
        A = adjacency_matrix(graph)

        n,n = size(A)
        L = scaled_laplacian(graph,scale)

        calc_sigma = opnorm(I-1/n*ones(n,n)-L)
        @printf("calculated sigma = %0.5f\n", calc_sigma)

        if varying_sigma == true
            if i ==1
                optparam, rho = optimize_parameters(sigma,kappa,false,[])
            else
                optparam, rho = optimize_parameters(sigma,kappa,true,prev_optparam)
            end
            alpha, delta, eta, zeta = optparam
        end

        # Make objective functions
        # Note, the output logistic_regression_COSMO_chip_data results in a random partitioning of the data
        # therefore, the output will be slightly different everytime it is called
        # TODO: To make the comparision fair, save the result using Serialize or by 
        # putting the code into jupyter notebook

        if problem_type == "LogisticRegression"
            f = x -> LogReg(x,H,Gam) # Calculate the cost
        end

        data,X,Y,H,Gam = opt_problem(n,6,0,1)
        X = X'
        d = size(X)[2]
        
        # Compiles the state matrices using the optimal parameters
        # Must weight the step size by the Lipschitz constant since 
        # LMI was solved wrt kappa
        
        G = [1 0 -alpha/(2*kappa/n) -zeta;
             1 1 0 -1;
             1 0 0 -1;
             delta eta 0 0]

        # Solve centralized problem using Mosek
        beta = Variable(d)
        problem = minimize(logisticloss(-Y.*(X*beta)) + sumsquares(beta))
        Convex.solve!(problem, () -> Mosek.Optimizer(QUIET = true, MSK_DPAR_BASIS_TOL_S = 1.0e-9, 
            MSK_DPAR_BASIS_TOL_X=1.0e-9,MSK_DPAR_INTPNT_CO_TOL_DFEAS=1.0e-15,
            MSK_DPAR_INTPNT_CO_TOL_MU_RED=1.0e-12,MSK_DPAR_INTPNT_CO_TOL_PFEAS=1.0e-12,
            MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1.0e-12),verbose=false)

        xopt = evaluate(beta)

        println(size(xopt))

        # Simulate SH-SVL with and without packet loss, x and xp are the
        # only signals we care about
        u,v,w,x,y,lossless_time = unstable_gd(G,f,d,L,iter;random_init=true,xopt=xopt)#;drop_prob=0.3*ones(size(L)),predict = eta)
        u,v,w,xp,y,h,lossy_time= unstable_gd(G,f,d,L,iter;random_init=true, drop_prob=pl*ones(size(L)),predict = eta,xopt=xopt)

        display(time)

        e = [maximum([norm(x[i,:,k].-xopt,Inf) for i=1:n]) for k=1:lossless_time]
        e2 = [maximum([norm(xp[i,:,k].-xopt,Inf) for i=1:n]) for k=1:lossy_time]

        rho1,rho2 = window_rho(e,e2,lossless_time,lossy_time)

        if optimize_params == true
            metadata[i,"opt_rho"] = rho  
        end
        metadata[i,"lossless_rho"] = rho1
        metadata[i,"lossy_rho"] = rho2
        metadata[i,"lossless_t"] = lossless_time
        metadata[i,"lossy_t"] = lossy_time
        metadata[i,"lossless_e"] = e[end]
        metadata[i,"lossy_e"] = e2[end]

        @printf("Alg 2 lossless rho = %0.4f\n", rho1)
        @printf("Alg 2 lossy rho = %0.4f\n", rho2)

        # Plot
        err = L"\max_i ||x_i-x_{opt}||_\infty" # using LaTeXStrings to embed math in the label
        plot(e, yaxis=:log,ylabel=err,xlabel="iterations",label="Alg 2, lossless",color=1,linestyle=:solid, size=[800,600], margin=10Measures.mm)
        plot!(e2, yaxis=:log,ylabel=err,xlabel="iterations",label="Alg 2, lossy",color=1,linestyle=:dash)

        savefig("figures/"*filename*"/"*name*".png")

        prev_optparam = Diagonal(optparam)

    end

    CSV.write("data/"*filename*".csv",metadata)
    
end


#simulate_file("n-custom_erdos_renyi-10-20-0.65opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
#simulate_file("n-custom_barabasi_albert-10-40-0.7opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)


# simulate_file("n-custom_barabasi_albert-10-50-0.7",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
# simulate_file("n-custom_watts_strogatz-10-50-0.7",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.7",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
# simulate_file("n-custom_geometric-10-50-0.7",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)

# simulate_file("n-custom_barabasi_albert-10-100-0.6opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_barabasi_albert-10-100-0.7opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_barabasi_albert-10-100-0.8opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_barabasi_albert-10-100-0.9opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)

# simulate_file("n-custom_barabasi_albert-10-50-0.65opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_barabasi_albert-10-50-0.8opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)

# simulate_file("n-custom_barabasi_albert-10-50-0.65",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_barabasi_albert-10-50-0.8",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)

# simulate_file("n-custom_erdos_renyi-10-50-0.2",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.35",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.5",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.65",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.8",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)

# simulate_file("n-custom_erdos_renyi-10-50-0.2opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.35opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.5opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.65opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_erdos_renyi-10-50-0.8opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)

# simulate_file("n-custom_geometric-10-50-0.2opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_geometric-10-50-0.35opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_geometric-10-50-0.5opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_geometric-10-50-0.65opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)
# simulate_file("n-custom_geometric-10-50-0.8opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,true,[],0.3)

# simulate_file("mu-custom_barabasi_albert-3-39-0.7",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
# simulate_file("mu-custom_erdos_renyi-3-39-0.7",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
# simulate_file("mu-custom_geometric-3-37-0.7",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)

# simulate_file("s-custom_barabasi_albert-0.2-0.95-15",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
# simulate_file("s-custom_erdos_renyi-0.2-0.95-15",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
# simulate_file("s-custom_geometric-0.2-0.95-15",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)

simulate_file("s-custom_barabasi_albert-0.2-0.95-15opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
simulate_file("s-custom_erdos_renyi-0.2-0.95-15opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)
simulate_file("s-custom_geometric-0.2-0.95-15opt",logistic_regression_COSMO_chip_data,"LogisticRegression",40000,true,false,[],0.3)




