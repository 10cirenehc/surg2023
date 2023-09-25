using Graphs, Karnak, Colors, NetworkLayout, StatsBase, Printf, Infiltrator, Integrals, Roots, LinearAlgebra, DataFrames, CSV
include("func_library.jl")

function draw(g)
    @drawsvg begin
        background("black")
        sethue("grey40")
        fontsize(8)
        drawgraph(g, 
            layout=stress, 
            vertexlabels = 1:nv(g),
            vertexfillcolors = 
                [RGB(rand(3)/2...) 
                for i in 1:nv(g)]
        )
    end 800 600
end

# Generates barabasi albert graph with target number of nodes n and 
# expected average degree mu
function custom_barabasi_albert(n,mu)
    m = Int32(round(n*mu/(2*(n-1))))
    # Create initial edgelist 
    el = Edge.(collect(zip(collect(1:m),collect(2:m+1))))
    # Create graph
    g = SimpleGraph(el)

    # add new vertices and connections to original nodes
    for node = m+2:n
        add_vertex!(g)
        d = degree(g)
        # sample targets of existing nodes to connect to
        targets = sample(collect(1:length(d)),Weights(d),m,replace=false)
        for t = targets
            add_edge!(g,node,t)
        end
    end
    
    return g
end

# Generates geometric on unit disk graph with target number of nodes n and 
# expected average degree mu
# https://mathworld.wolfram.com/DiskLinePicking.html
function custom_geometric(n,mu)
    fa(s,p) = ((4/pi)*s.*acos(s/2)-(2/pi)*(s.^2).*sqrt(1-(s/2).^2));
    fb(r) = ((n-1)*Roots.solve(IntegralProblem(fa,0.0,r), HCubatureJL()).-mu);
    R2 = fzero(fb,(0,2));
    r = sqrt.(rand(1,n));
    theta = 2*pi*rand(1,n);
    D = r.^2 .+ r'.^2 - 2*r.*r'.*cos.(theta.-theta');
    A = D .<= R2^2;
    A[diagind(A)] .= 0;
    g = SimpleGraph(A);

    return g
end

function custom_erdos_renyi(n,mu)
    p = mu/(n-1);
    A = rand(Float64,(n,n)) .<=p
    A[diagind(A)] .= 0;
    
    g = SimpleGraph(UpperTriangular(A') .+ LowerTriangular(A));
    return g
end

function custom_watts_strogatz(n,mu)
    nu = ceil(mu/2);
    g = SimpleGraph(n);
    
    for k = 1:n
        for j = 1:nu
            add_edge!(g,k,mod(k+j-1,n)+1);
        end
    end

    beta = 0.01;
    for k = 1:n
        for t = k:k-1+nu
            if rand(1)[1] <= beta
                p = rand(1:n);
                while k==p || in(p, neighbors(g,k))
                    p = rand(1:n)
                end
                rem_edge!(g,k,Int32(mod(t,n)+1))
                add_edge!(g,k,p)
            end
        end
    end

    return g

end

function generate_var_n(func::Function, min_n::Int64, max_n::Int64, number::Int64, mu_percent::Float64, 
tolerance::Float64, optimize::Bool, weight::Float64=1.0,sigma::Float64=0.0)

    interval = round((max_n-min_n)/number)
    graph_dict = Dict{String,SimpleGraph}()
    metadata = DataFrame(name=[],n=[],mu=[],target=[],sigma=[],sigma_error=[],scale=[],type=[])
    
    for n = min_n:interval:max_n

        # match to target sigma
        if optimize == true
            mu = round(n*mu_percent)
            if func==custom_geometric
                mu_s = 1:Int64(ceil(mu/15)):n-3
            else
                mu_s = 1:Int64(ceil(mu/15)):n-1
            end

            graphs = func.(Ref(Int64(n)),Int64.(mu_s))
            laplacians = scaled_laplacian.(graphs,Ref(weight))
            results = spectral_gap.(laplacians)
            sigmas = getfield.(results,2)
            lambdas = getfield.(results,1)
            
            # filter acceptable sigmas

            sigma_errors = abs.(sigmas .- sigma)/sigma
            mask = sigma_errors .< tolerance

            sigmas = sigmas[mask]
            graphs = graphs[mask]
            mu_s = mu_s[mask]
            sigma_errors = sigma_errors[mask]
            lambdas = lambdas[mask]
            
            if length(sigmas) == 0
                println("No acceptable graphs found for n = " * string(n))
                continue
            end
            
            for i in eachindex(graphs)
                name = "g"*string(n)*"_" * string(mu_s[i])
                graph_dict[name] = graphs[i]
                push!(metadata, [name, n, mu_s[i], sigma,sigmas[i], sigma_errors[i], lambdas[i], string(func)])
            end

        else
            # multiply the Laplacian by a constant to get different sigmas
            mu = round(n*mu_percent)
            g = func(Int64(n),Int64(mu));
            name = "g"*string(n)
            scales = 0.01:0.0025:1
            laplacians = scaled_laplacian.(Ref(g),scales)
            sigmas = basic_sigma.(laplacians)

            sigma_errors = abs.(sigmas .- sigma)/sigma
            min_error, index = findmin(sigma_errors)

            if min_error > tolerance
                println("No acceptable graphs found for n = " * string(n))
            else
                push!(metadata, [name, n, mu, sigma, sigmas[index], min_error, scales[index], string(func)])
            end
            
            graph_dict[name] = g
        end
        
    end

    if optimize == true
        savegraph("data/n-"*string(func)*"-"*string(min_n)*"-"*string(max_n)*"-"*string(sigma)*"opt.lg",graph_dict)
        CSV.write("data/n-"*string(func)*"-"*string(min_n)*"-"*string(max_n)*"-"*string(sigma)*"opt.csv", metadata)
    else
        savegraph("data/n-"*string(func)*"-"*string(min_n)*"-"*string(max_n)*"-"*string(sigma)*".lg",graph_dict)
        CSV.write("data/n-"*string(func)*"-"*string(min_n)*"-"*string(max_n)*"-"*string(sigma)*".csv", metadata)
    end
end

function generate_var_mu(func::Function, min_mu::Int64, max_mu::Int64, number::Int64, n::Int64, 
    tolerance::Float64, optimize::Bool, weight::Float64=1.0,sigma::Float64=0.0)
    
        interval = round((max_mu-min_mu)/number)
        graph_dict = Dict{String,SimpleGraph}()
        metadata = DataFrame(name=[],n=[],mu=[],target=[],sigma=[],sigma_error=[],scale = [], type=[])
        
        for mu = min_mu:interval:max_mu
    
            # match to target sigma
            if optimize == true
                n_s = (mu+2):ceil(n/20):n
                graphs = func.(Int64.(n_s),Ref(Int64(mu)))
                laplacians = scaled_laplacian.(graphs,Ref(weight))
                results = spectral_gap.(laplacians)
                sigmas = getfield.(results,2)
                lambdas = getfield.(results,1)

                
                # filter acceptable sigmas
    
                sigma_errors = abs.(sigmas .- sigma)/sigma
                mask = sigma_errors .< tolerance
    
                sigmas = sigmas[mask]
                graphs = graphs[mask]
                n_s = n_s[mask]
                sigma_errors = sigma_errors[mask]
                lambdas = lambdas[mask]

                if length(sigmas) == 0
                    println("No acceptable graphs found for n = " * string(n))
                end
                
                for i in eachindex(graphs)
                    name = "g"*string(n)*"_" * string(mu_s[i])
                    graph_dict[name] = graphs[i]
                    push!(metadata, [name, n_s[i], mu, sigma,sigmas[i], sigma_errors[i], lambdas[i], string(func)])
                end
    
            else
                # multiply the Laplacian by a constant to get different sigmas
                g = func(n,Int64(mu));
                name = "g"*string(mu)
                println(mu)
                scales = 0.01:0.00251:1
                laplacians = scaled_laplacian.(Ref(g),scales)
                sigmas = basic_sigma.(laplacians)

                sigma_errors = abs.(sigmas .- sigma)/sigma
                min_error, index = findmin(sigma_errors)

                if min_error > tolerance
                    println("No acceptable graphs found for n = " * string(n))
                else
                    push!(metadata, [name, n, mu, sigma, sigmas[index], min_error, scales[index], string(func)])
                end
                
                graph_dict[name] = g
            end
            
        end
        
        if optimize == true
            savegraph("data/mu-"*string(func)*"-"*string(min_mu)*"-"*string(max_mu)*"-"*string(sigma)*"opt.lg",graph_dict)
            CSV.write("data/mu-"*string(func)*"-"*string(min_mu)*"-"*string(max_mu)*"-"*string(sigma)*"opt.csv", metadata)
        else
            savegraph("data/mu-"*string(func)*"-"*string(min_mu)*"-"*string(max_mu)*"-"*string(sigma)*".lg",graph_dict)
            CSV.write("data/mu-"*string(func)*"-"*string(min_mu)*"-"*string(max_mu)*"-"*string(sigma)*".csv", metadata)
        end
    end

function generate_var_n(func::Function, min_n::Int64, max_n::Int64, number::Int64, mu_percent::Float64, 
tolerance::Float64, optimize::Bool, weight::Float64=1.0,sigma::Float64=0.0)

    interval = round((max_n-min_n)/number)
    graph_dict = Dict{String,SimpleGraph}()
    metadata = DataFrame(name=[],n=[],mu=[],target=[],sigma=[],sigma_error=[],scale=[],type=[])
    
    for n = min_n:interval:max_n

        # match to target sigma
        if optimize == true
            mu = round(n*mu_percent)
            if func==custom_geometric
                mu_s = 1:Int64(ceil(mu/15)):n-3
            else
                mu_s = 1:Int64(ceil(mu/15)):n-1
            end

            graphs = func.(Ref(Int64(n)),Int64.(mu_s))
            laplacians = scaled_laplacian.(graphs,Ref(weight))
            results = spectral_gap.(laplacians)
            sigmas = getfield.(results,2)
            lambdas = getfield.(results,1)
            
            # filter acceptable sigmas

            sigma_errors = abs.(sigmas .- sigma)/sigma
            mask = sigma_errors .< tolerance

            sigmas = sigmas[mask]
            graphs = graphs[mask]
            mu_s = mu_s[mask]
            sigma_errors = sigma_errors[mask]
            lambdas = lambdas[mask]
            
            if length(sigmas) == 0
                println("No acceptable graphs found for n = " * string(n))
                continue
            end
            
            for i in eachindex(graphs)
                name = "g"*string(n)*"_" * string(mu_s[i])
                graph_dict[name] = graphs[i]
                push!(metadata, [name, n, mu_s[i], sigma,sigmas[i], sigma_errors[i], lambdas[i], string(func)])
            end

        else
            # multiply the Laplacian by a constant to get different sigmas
            mu = round(n*mu_percent)
            g = func(Int64(n),Int64(mu));
            name = "g"*string(n)
            scales = 0.01:0.0025:1
            laplacians = scaled_laplacian.(Ref(g),scales)
            sigmas = basic_sigma.(laplacians)

            sigma_errors = abs.(sigmas .- sigma)/sigma
            min_error, index = findmin(sigma_errors)

            if min_error > tolerance
                println("No acceptable graphs found for n = " * string(n))
            else
                push!(metadata, [name, n, mu, sigma, sigmas[index], min_error, scales[index], string(func)])
            end
            
            graph_dict[name] = g
        end
        
    end

    if optimize == true
        savegraph("data/n-"*string(func)*"-"*string(min_n)*"-"*string(max_n)*"-"*string(sigma)*"opt.lg",graph_dict)
        CSV.write("data/n-"*string(func)*"-"*string(min_n)*"-"*string(max_n)*"-"*string(sigma)*"opt.csv", metadata)
    else
        savegraph("data/n-"*string(func)*"-"*string(min_n)*"-"*string(max_n)*"-"*string(sigma)*".lg",graph_dict)
        CSV.write("data/n-"*string(func)*"-"*string(min_n)*"-"*string(max_n)*"-"*string(sigma)*".csv", metadata)
    end
end

function generate_var_sigma(func::Function, min_s::Float64, max_s::Float64, number::Int64, n::Int64, mu_percent::Float64, 
    tolerance::Float64, optimize::Bool)
    
        interval = (max_s-min_s)/number
        graph_dict = Dict{String,SimpleGraph}()
        metadata = DataFrame(name=[],n=[],mu=[],target=[],sigma=[],sigma_error=[],scale=[],type=[])
        
        for sigma = min_s:interval:max_s
    
            # match to target sigma

            if optimize == true
                if func==custom_geometric
                    mu_s = 1:1:n-3
                else
                    mu_s = 1:1:n-1
                end

                graphs = func.(Ref(Int64(n)),Int64.(mu_s))
                laplacians = scaled_laplacian.(graphs,1.0)
                results = spectral_gap.(laplacians)
                sigmas = getfield.(results,2)
                lambdas = getfield.(results,1)
                
                # filter acceptable sigmas

                sigma_errors = abs.(sigmas .- sigma)/sigma
                mask = sigma_errors .< tolerance

                sigmas = sigmas[mask]
                graphs = graphs[mask]
                mu_s = mu_s[mask]
                sigma_errors = sigma_errors[mask]
                lambdas = lambdas[mask]
                
                if length(sigmas) == 0
                    println("No acceptable graphs found for sigma = " * string(sigma))
                    continue
                end
                
                for i in eachindex(graphs)
                    name = "g"*string(sigma)*"_" * string(mu_s[i])
                    graph_dict[name] = graphs[i]
                    push!(metadata, [name, n, mu_s[i], sigma,sigmas[i], sigma_errors[i], lambdas[i], string(func)])
                end

                
            else
                # multiply the Laplacian by a constant to get different sigmas
                mu = round(n*mu_percent)
                g = func(Int64(n),Int64(mu));
                name = "g"*string(sigma)
                scales = 0.01:0.001:1
                laplacians = scaled_laplacian.(Ref(g),scales)
                sigmas = basic_sigma.(laplacians)
    
                sigma_errors = abs.(sigmas .- sigma)/sigma
                min_error, index = findmin(sigma_errors)
    
                if min_error > tolerance
                    println("No acceptable graphs found for sigma = " * string(sigma))
                else
                    push!(metadata, [name, n, mu, sigma, sigmas[index], min_error, scales[index], string(func)])
                end
                
                graph_dict[name] = g
            end
            
        end

        if optimize == true
            savegraph("data/s-"*string(func)*"-"*string(min_s)*"-"*string(max_s)*"-"*string(n)*"opt.lg",graph_dict)
            CSV.write("data/s-"*string(func)*"-"*string(min_s)*"-"*string(max_s)*"-"*string(n)*"opt.csv", metadata)
        else
            savegraph("data/s-"*string(func)*"-"*string(min_s)*"-"*string(max_s)*"-"*string(n)*".lg",graph_dict)
            CSV.write("data/s-"*string(func)*"-"*string(min_s)*"-"*string(max_s)*"-"*string(n)*".csv", metadata)
        end
        
    end

#generate_var_n(custom_barabasi_albert,10,20,10,9,0.1,true,0.25,0.85)
#generate_var_n(custom_barabasi_albert,10,20,10,9,0.1,false,0.25,0.85)
#custom_erdos_renyi(10,5)

#generate_var_n(custom_erdos_renyi,10,20,3,9,0.1,true,1.0,0.65)
#generate_var_n(custom_barabasi_albert,10,40,15,9,0.1,true,1.0,0.70)












