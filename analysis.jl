using DataFrames, CSV, Plots, Statistics, LaTeXStrings

include("func_library.jl")

function varying_mu()
    df1 = DataFrame(CSV.File("data/mu-custom_barabasi_albert-3-39-0.7.csv",header=true))
    df2 = DataFrame(CSV.File("data/mu-custom_erdos_renyi-3-39-0.7.csv",header=true))
    df3 = DataFrame(CSV.File("data/mu-custom_geometric-3-37-0.7.csv",header=true))

    plot(df1.mu,df1.lossless_rho, label = "Barabasi Albert")
    plot!(df2.mu,df2.lossless_rho, label = "Erdos Renyi")
    plot!(df3.mu,df3.lossless_rho, label = "Geometric")
    xlabel!("average degree")
    ylabel!("lossless rho")
    savefig("figures/plots/mu.png")

    plot(df1.mu,df1.lossy_rho, label = "Barabasi Albert")
    plot!(df2.mu,df2.lossy_rho, label = "Erdos Renyi")
    plot!(df3.mu,df3.lossy_rho, label = "Geometric")
    xlabel!("average degree")
    ylabel!("lossy rho")
    savefig("figures/plots/mulossy.png")
end 

# Comparing Four types of graphs
function compare_types()
    df1 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-50-0.7.csv",header=true))
    df2 = DataFrame(CSV.File("data/n-custom_erdos_renyi-10-50-0.7.csv",header=true))
    df3 = DataFrame(CSV.File("data/n-custom_geometric-10-50-0.7.csv",header=true))
    df4 = DataFrame(CSV.File("data/n-custom_watts_strogatz-10-50-0.7.csv",header=true))

    plot(df1.n,df1.lossless_rho, label = "Barabasi Albert", legend= :right)
    plot!(df2.n,df2.lossless_rho, label = "Erdos Renyi")
    plot!(df3.n,df3.lossless_rho, label = "Geometric")
    plot!(df4.n,df4.lossless_rho, label = "Watts Strogatz")
    xlabel!("number of nodes")
    ylabel!("lossless rho")
    savefig("figures/plots/4types.png")

    plot(df1.n,df1.lossy_rho, label = "Barabasi Albert", legend= :outerright)
    plot!(df2.n,df2.lossy_rho, label = "Erdos Renyi")
    plot!(df3.n,df3.lossy_rho, label = "Geometric")
    plot!(df4.n,df4.lossy_rho, label = "Watts Strogatz")
    xlabel!("number of nodes")
    ylabel!("lossy rho")
    savefig("figures/plots/4typeslossy.png")

    df1_lossless,df1_lossy = calc_ratio(df1.opt_rho,df1.lossless_rho,df1.lossy_rho)

    println(mean(df1_lossless))
    println(mean(df1_lossy))

    df2_lossless,df2_lossy = calc_ratio(df2.opt_rho,df2.lossless_rho,df2.lossy_rho)

    println(mean(df2_lossless))
    println(mean(df2_lossy))

    df3_lossless,df3_lossy = calc_ratio(df3.opt_rho,df3.lossless_rho,df3.lossy_rho)

    println(mean(df3_lossless))
    println(mean(df3_lossy))

    df4_lossless,df4_lossy = calc_ratio(df4.opt_rho,df4.lossless_rho,df4.lossy_rho)

    println(mean(df4_lossless))
    println(mean(df4_lossy))
end

# # Varying sigma
function vary_sigma_1()
    df1 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-100-0.6opt.csv",header=true))
    df2 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-100-0.7opt.csv",header=true))
    df3 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-100-0.8opt.csv",header=true))
    df4 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-100-0.9opt.csv",header=true))

    plot(df2.n, [df2.opt_rho, df2.lossless_rho, df2.lossy_rho], label=["theoretical rho" "lossless rho" "lossy rho"])

    xlabel!("Number of nodes")
    ylabel!("Rho")

    df1_lossless,df1_lossy = calc_ratio(df1.opt_rho,df1.lossless_rho,df1.lossy_rho)

    println(mean(df1_lossless))
    println(mean(df1_lossy))
end


# Varying sigma
function varying_sigma_2()

    # First remove duplicates from the optimized sigma graphs
    remove_duplicates("data/s-custom_erdos_renyi-0.2-0.95-15opt.csv","target")
    remove_duplicates("data/s-custom_geometric-0.2-0.95-15opt.csv","target")
    remove_duplicates("data/s-custom_barabasi_albert-0.2-0.95-15opt.csv","target")

    er_1 = DataFrame(CSV.File("data/s-custom_erdos_renyi-0.2-0.95-15.csv",header=true))
    er_2 = DataFrame(CSV.File("data/s-custom_erdos_renyi-0.2-0.95-15opt.csv",header=true))

    geo_1 = DataFrame(CSV.File("data/s-custom_geometric-0.2-0.95-15.csv",header=true))
    geo_2 = DataFrame(CSV.File("data/s-custom_geometric-0.2-0.95-15opt.csv",header=true))

    ba_1 = DataFrame(CSV.File("data/s-custom_barabasi_albert-0.2-0.95-15.csv",header=true))
    ba_2 = DataFrame(CSV.File("data/s-custom_barabasi_albert-0.2-0.95-15opt.csv",header=true))

    # First plot
    plot(er_1.sigma, er_1.opt_rho, label="optimized worst-case", legend = :outerbottom)
    scatter!(er_1.sigma, [er_1.lossless_rho,er_1.lossy_rho], label = ["lossless" "lossy"])
    plot!(er_1.sigma, theoretical_lower_bound(er_1.kappa,er_1.sigma), label=["theoretical lower bound"], linestyle=:dash)
    xlabel!("Sigma")
    ylabel!("Convergence rate")
    title!("Erdos-Renyi (unoptimized sigma)")
    savefig("figures/plots/s-er.png")

    plot(geo_1.sigma, geo_1.opt_rho, label="optimized worst-case")
    scatter!(geo_1.sigma, [geo_1.lossless_rho,geo_1.lossy_rho], label = ["lossless" "lossy"])
    plot!(geo_1.sigma, theoretical_lower_bound(geo_1.kappa,geo_1.sigma), label=["theoretical lower bound"], linestyle=:dash)
    xlabel!("Sigma")
    ylabel!("Convergence rate")
    title!("Geometric (unoptimized sigma)")
    savefig("figures/plots/s-geo.png")

    plot(ba_1.sigma, ba_1.opt_rho, label="optimized worst-case")
    scatter!(ba_1.sigma, [ba_1.lossless_rho,ba_1.lossy_rho], label = ["lossless" "lossy"])
    plot!(ba_1.sigma, theoretical_lower_bound(ba_1.kappa,ba_1.sigma), label=["theoretical lower bound"],linestyle=:dash)
    xlabel!("Sigma")
    ylabel!("Convergence rate")
    title!("Barabasi Albert (unoptimized sigma)")
    savefig("figures/plots/s-ba.png")

    er_1_lossless, er_1_lossy = calc_ratio(er_1.opt_rho, er_1.lossless_rho, er_1.lossy_rho)
    ba_1_lossless, ba_1_lossy = calc_ratio(ba_1.opt_rho, ba_1.lossless_rho, ba_1.lossy_rho)
    geo_1_lossless, geo_1_lossy = calc_ratio(geo_1.opt_rho, geo_1.lossless_rho, geo_1.lossy_rho)

    plot(er_2.sigma, er_2.opt_rho, label="optimized worst-case")
    scatter!(er_2.sigma, [er_2.lossless_rho,er_2.lossy_rho], label = ["lossless" "lossy"])
    plot!(er_2.sigma, theoretical_lower_bound(er_2.kappa,er_2.sigma), label=["theoretical lower bound"], linestyle=:dash)
    xlabel!("Sigma")
    ylabel!("Convergence rate")
    title!("Erdos-Renyi (optimized sigma)")
    savefig("figures/plots/s-er-opt.png")

    plot(geo_2.sigma, geo_2.opt_rho, label="optimized worst-case")
    scatter!(geo_2.sigma, [geo_2.lossless_rho,geo_2.lossy_rho], label = ["lossless" "lossy"])
    plot!(geo_2.sigma, theoretical_lower_bound(geo_2.kappa,geo_2.sigma), label=["theoretical lower bound"], linestyle=:dash)
    xlabel!("Sigma")
    ylabel!("Convergence rate")
    title!("Geometric (optimized sigma)")
    savefig("figures/plots/s-geo-opt.png")

    plot(ba_2.sigma, ba_2.opt_rho, label="optimized worst-case")
    scatter!(ba_2.sigma, [ba_2.lossless_rho,ba_2.lossy_rho], label = ["lossless" "lossy"])
    plot!(ba_2.sigma, theoretical_lower_bound(ba_2.kappa,ba_2.sigma), label=["theoretical lower bound"],linestyle=:dash)
    xlabel!("Sigma")
    ylabel!("Convergence rate")
    title!("Barabasi Albert (optimized sigma)")
    savefig("figures/plots/s-ba-opt.png")

    # Distributions of log ratio

    scatter(er_1.sigma, [er_1_lossless,er_1_lossy],label=["lossless" "lossy"])
    xlabel!("Sigma")
    ylabel = L"\frac{{log(\rho_{sim})}}{{log(\rho_{worst-case})}}"
    ylabel!(ylabel)
    title!("Erdos Renyi")
    savefig("figures/plots/s-er-ratio.png")

    scatter(geo_1.sigma, [geo_1_lossless,geo_1_lossy],label=["lossless" "lossy"])
    xlabel!("Sigma")
    ylabel = L"\frac{{log(\rho_{sim})}}{{log(\rho_{worst-case})}}"
    ylabel!(ylabel)
    title!("Geometric")
    savefig("figures/plots/s-geo-ratio.png")

    scatter(ba_1.sigma, [ba_1_lossless,ba_1_lossy],label=["lossless" "lossy"])
    xlabel!("Sigma")
    ylabel = L"\frac{{log(\rho_{sim})}}{{log(\rho_{worst-case})}}"
    ylabel!(ylabel)
    title!("Barabasi Albert")
    savefig("figures/plots/s-ba-ratio.png")

    println(mean(er_1_lossless))
    println(mean(er_1_lossy))
    println(mean(geo_1_lossless))
    println(mean(geo_1_lossy))
    println(mean(ba_1_lossless))
    println(mean(ba_1_lossy))



    # plot(er_2.sigma, er_2.opt_rho, label=["theoretical rho"])
    # scatter!(er_2.sigma, [er_2.lossless_rho,er_2.lossy_rho], label = ["lossless" "lossy"])
    # xlabel!("Sigma")
    # ylabel!("Convergence rate")
    # title!("Erdos-Renyi (optimized sigma)")
    # savefig("figures/plots/s-er-opt.png")

end

function vary_packet_loss()
    packet_losses = [0.1 0.3 0.5 0.7 0.9]
    data = [] 
    
    # Plot of convergence versus sigma
    df0 = DataFrame(CSV.File("data/s-custom_erdos_renyi-0.2-0.95-15.csv",header=true))
    plot(df0.sigma, df0.opt_rho, label="optimized worst-case",dpi=1000)
    scatter!(df0.sigma, [df0.lossless_rho], label = ["lossless"],markersize=2)
    for i = 1:length(packet_losses)
        df = DataFrame(CSV.File("data/s-custom_erdos_renyi-0.2-0.95-15$(packet_losses[i]).csv",header=true))
        scatter!(df.sigma, [df.lossy_rho], label = ["pl = $(packet_losses[i])"], markersize=2)
    end

    xlabel!("Sigma")
    ylabel!("Convergence rate")
    title!("Erdos Renyi with varying packet loss")
    savefig("figures/plots/s-er-pl.png")

    indices = [4, 19, 36, 47]
    indices2 = [2,10,18,26,34,48]

    packet_losses = [0.1:0.05:0.95; 0.98]

    plot_times = zeros(length(packet_losses)+1,4)
    plot_rates = zeros(length(packet_losses)+1,4)
    plot_ratios = zeros(length(packet_losses)+1,4)
    plot_times2 = zeros(length(packet_losses)+1,6)

    for i = 1:length(packet_losses)
        df = DataFrame(CSV.File("data/s-custom_erdos_renyi-0.2-0.95-15$(packet_losses[i]).csv",header=true))
        push!(data,df)
    end

    plot_times[1,:] = df0.lossless_t[indices]
    plot_rates[1,:] = df0.lossless_rho[indices]
    plot_ratios[1,:] = calc_ratio(df0.opt_rho[indices],df0.lossless_rho[indices],df0.lossy_rho[indices])[1]
    plot_times2[1,:] = df0.lossless_t[indices2]
    for i = 1:length(data)
        for j = 1:length(indices)
            plot_times[i+1,j] = data[i].lossy_t[indices[j]]
            plot_rates[i+1,j] = data[i].lossy_rho[indices[j]]
            plot_ratios[i+1,j] = calc_ratio(data[i].opt_rho[indices[j]],data[i].lossless_rho[indices[j]],data[i].lossy_rho[indices[j]])[2]
        end
    end
    for i = 1:length(data)
        for j = 1:length(indices2)
            plot_times2[i+1,j] = data[i].lossy_t[indices2[j]]
        end
    end

    # Plot convergence time for a sigmas 0.26, 0.5, 0.75, 0.92
    packet_losses = [0;packet_losses]
    plot(packet_losses, plot_times[:,1], label="sigma = 0.26",dpi=1000)
    plot!(packet_losses, plot_times[:,2], label="sigma = 0.5")
    plot!(packet_losses, plot_times[:,3], label="sigma = 0.75")
    plot!(packet_losses, plot_times[:,4], label="sigma = 0.92")
    xlabel!("Packet loss")
    ylabel!("Convergence time")
    title!("Convergence Iterations with varying packet loss")
    savefig("figures/plots/s-er-pl-time.png")

    # Plot convergence time for a sigmas 0.215, 0.37, 0.485, 0.605, 0.725
    plot(packet_losses, plot_times2[:,1], label="sigma = 0.215",dpi=1000)
    plot!(packet_losses, plot_times2[:,2], label="sigma = 0.37")
    plot!(packet_losses, plot_times2[:,3], label="sigma = 0.485")
    plot!(packet_losses, plot_times2[:,4], label="sigma = 0.605")
    plot!(packet_losses, plot_times2[:,5], label="sigma = 0.725")
    plot!(packet_losses, plot_times2[:,5], label="sigma = 0.935")

    xlabel!("Packet loss")
    ylabel!("Convergence time")
    title!("Convergence Iterations with varying packet loss")
    savefig("figures/plots/s-er-pl-time_v2.png")

    # Plot convergence rate for a sigmas 0.26, 0.5, 0.75, 0.92
    plot(packet_losses, plot_rates[:,1], label="sigma = 0.26",dpi=1000)
    plot!(packet_losses, plot_rates[:,2], label="sigma = 0.5")
    plot!(packet_losses, plot_rates[:,3], label="sigma = 0.75")
    plot!(packet_losses, plot_rates[:,4], label="sigma = 0.92")
    xlabel!("Packet loss")
    ylabel!("Convergence rate")
    title!("Convergence rate with varying packet loss")
    savefig("figures/plots/s-er-pl-rate.png")

    # Plot ratio for a sigmas 0.26, 0.5, 0.75, 0.92
    plot(packet_losses, plot_ratios[:,1], label="sigma = 0.26",dpi=1000)
    plot!(packet_losses, plot_ratios[:,2], label="sigma = 0.5")
    plot!(packet_losses, plot_ratios[:,3], label="sigma = 0.75")
    plot!(packet_losses, plot_ratios[:,4], label="sigma = 0.92")
    xlabel!("Packet loss")
    ylabel!("log(sim_rho)/log(worst-case_rho)")
    title!("Ratio with varying packet loss")
    savefig("figures/plots/s-er-pl-ratio.png")

    # scatter of first timeout for each sigma
    sigmas = df0.sigma
    thresholds = []
    for i = 1:length(sigmas)
        flag = false
        for j = 1:length(data)
            if data[j].lossy_t[i] == 40000
                push!(thresholds,packet_losses[j+1])
                flag = true
                break
            end
        end
        if flag == false
            push!(thresholds,1)
        end
    end

    plot(sigmas,thresholds,label = "Packet loss probability",dpi=1000)
    xlabel!("Sigma")
    ylabel!("Packet loss threshold")
    title!("Lowest package loss at which simulation times out (40000 iterations)", titlefontsize=8)
    savefig("figures/plots/s-er-pl-threshold.png")
    
end



