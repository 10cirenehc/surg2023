using DataFrames, CSV, Plots, Statistics

include("func_library.jl")

# # Four types of graphs

# df1 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-50-0.7.csv",header=true))
# df2 = DataFrame(CSV.File("data/n-custom_erdos_renyi-10-50-0.7.csv",header=true))
# df3 = DataFrame(CSV.File("data/n-custom_geometric-10-50-0.7.csv",header=true))
# df4 = DataFrame(CSV.File("data/n-custom_watts_strogatz-10-50-0.7.csv",header=true))

# plot(df1.n,df1.lossless_rho, label = "Barabasi Albert")
# plot!(df2.n,df2.lossless_rho, label = "Erdos Renyi")
# plot!(df3.n,df3.lossless_rho, label = "Geometric")
# plot!(df4.n,df4.lossless_rho, label = "Watts Strogatz")
# xlabel!("number of nodes")
# ylabel!("lossless rho")
# savefig("figures/plots/4types.png")

# plot(df1.n,df1.lossy_rho, label = "Barabasi Albert")
# plot!(df2.n,df2.lossy_rho, label = "Erdos Renyi")
# plot!(df3.n,df3.lossy_rho, label = "Geometric")
# plot!(df4.n,df4.lossy_rho, label = "Watts Strogatz")
# xlabel!("number of nodes")
# ylabel!("lossless rho")
# savefig("figures/plots/4typeslossy.png")

# df1_lossless,df1_lossy = calc_ratio(df1.opt_rho,df1.lossless_rho,df1.lossy_rho)

# println(mean(df1_lossless))
# println(mean(df1_lossy))

# df2_lossless,df2_lossy = calc_ratio(df2.opt_rho,df2.lossless_rho,df2.lossy_rho)

# println(mean(df2_lossless))
# println(mean(df2_lossy))

# df3_lossless,df3_lossy = calc_ratio(df3.opt_rho,df3.lossless_rho,df3.lossy_rho)

# println(mean(df3_lossless))
# println(mean(df3_lossy))

# df4_lossless,df4_lossy = calc_ratio(df4.opt_rho,df4.lossless_rho,df4.lossy_rho)

# println(mean(df4_lossless))
# println(mean(df4_lossy))

# # Varying sigma

# df1 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-100-0.6opt.csv",header=true))
# df2 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-100-0.7opt.csv",header=true))
# df3 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-100-0.8opt.csv",header=true))
# df4 = DataFrame(CSV.File("data/n-custom_barabasi_albert-10-100-0.9opt.csv",header=true))

# plot(df2.n, [df2.opt_rho, df2.lossless_rho, df2.lossy_rho], label=["theoretical rho" "lossless rho" "lossy rho"])

# xlabel!("Number of nodes")
# ylabel!("Rho")

# df1_lossless,df1_lossy = calc_ratio(df1.opt_rho,df1.lossless_rho,df1.lossy_rho)

# println(mean(df1_lossless))
# println(mean(df1_lossy))



# Varying sigma

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
plot(er_1.sigma, er_1.opt_rho, label="optimized worst-case")
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
ylabel!("log(sim_rho)/log(worst-case_rho)")
title!("Erdos Renyi")
savefig("figures/plots/s-er-ratio.png")

scatter(geo_1.sigma, [geo_1_lossless,geo_1_lossy],label=["lossless" "lossy"])
xlabel!("Sigma")
ylabel!("log(sim_rho)/log(worst-case_rho)")
title!("Geometric")
savefig("figures/plots/s-geo-ratio.png")

scatter(ba_1.sigma, [ba_1_lossless,ba_1_lossy],label=["lossless" "lossy"])
xlabel!("Sigma")
ylabel!("log(sim_rho)/log(worst-case_rho)")
title!("Barabasi Albert")
savefig("figures/plots/s-ba-ratio.png")

# plot(er_2.sigma, er_2.opt_rho, label=["theoretical rho"])
# scatter!(er_2.sigma, [er_2.lossless_rho,er_2.lossy_rho], label = ["lossless" "lossy"])
# xlabel!("Sigma")
# ylabel!("Convergence rate")
# title!("Erdos-Renyi (optimized sigma)")
# savefig("figures/plots/s-er-opt.png")




