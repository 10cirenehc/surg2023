using Graphs, Karnak, Colors, NetworkLayout, StatsBase, Printf, Infiltrator, Integrals, Roots, LinearAlgebra, DataFrames, CSV

include("func_library.jl")
include("graph_generation.jl")

# generate_var_n(custom_watts_strogatz,10,50,20,0.7,0.025,false,1.0,0.70)
# generate_var_n(custom_barabasi_albert,10,50,20,0.7,0.025,false,1.0,0.70)
# generate_var_n(custom_erdos_renyi,10,50,20,0.7,0.025,false,1.0,0.70)
# generate_var_n(custom_geometric,10,50,20,0.7,0.025,false,1.0,0.70)

# generate_var_n(custom_barabasi_albert,10,100,25,0.7,0.025,true,1.0,0.50)
# generate_var_n(custom_barabasi_albert,10,100,25,0.7,0.025,true,1.0,0.60)
# generate_var_n(custom_barabasi_albert,10,100,25,0.7,0.025,true,1.0,0.70)
# generate_var_n(custom_barabasi_albert,10,100,25,0.7,0.025,true,1.0,0.80)
# generate_var_n(custom_barabasi_albert,10,100,25,0.7,0.025,true,1.0,0.90)

# generate_var_n(custom_barabasi_albert,10,50,25,0.8,0.025,true,1.0,0.20)
# generate_var_n(custom_barabasi_albert,10,50,25,0.8,0.025,true,1.0,0.35)
# generate_var_n(custom_barabasi_albert,10,50,25,0.8,0.025,true,1.0,0.50)
# generate_var_n(custom_barabasi_albert,10,50,25,0.8,0.025,true,1.0,0.65)
# generate_var_n(custom_barabasi_albert,10,50,25,0.8,0.025,true,1.0,0.80)

# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,false,1.0,0.20)
# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,false,1.0,0.35)
# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,false,1.0,0.50)
# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,false,1.0,0.65)
# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,false,1.0,0.80)

# generate_var_n(custom_geometric,10,50,25,0.8,0.025,true,1.0,0.20)
# generate_var_n(custom_geometric,10,50,25,0.8,0.025,true,1.0,0.35)
# generate_var_n(custom_geometric,10,50,25,0.8,0.025,true,1.0,0.50)
# generate_var_n(custom_geometric,10,50,25,0.8,0.025,true,1.0,0.65)
# generate_var_n(custom_geometric,10,50,25,0.8,0.025,true,1.0,0.80)

# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,true,1.0,0.20)
# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,true,1.0,0.35)
# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,true,1.0,0.50)
# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,true,1.0,0.65)
# generate_var_n(custom_erdos_renyi,10,50,25,0.8,0.025,true,1.0,0.80)

# generate_var_mu(custom_erdos_renyi,3,39,15,40,0.025,false,1.0,0.70)
# generate_var_mu(custom_geometric,3,37,15,40,0.025,false,1.0,0.70)
# generate_var_mu(custom_barabasi_albert,3,39,15,40,0.025,false,1.0,0.70)

# generate_var_sigma(custom_erdos_renyi,0.2,0.95,50,15,0.8,0.025, false)
# generate_var_sigma(custom_geometric,0.2,0.95,50,50,0.8,0.025, false)
# generate_var_sigma(custom_barabasi_albert,0.2,0.95,50,15,0.8,0.025, false)

generate_var_sigma(custom_erdos_renyi,0.2,0.95,50,15,0.8,0.025, true)
generate_var_sigma(custom_geometric,0.2,0.95,50,15,0.8,0.025, true)
generate_var_sigma(custom_barabasi_albert,0.2,0.95,50,15,0.8,0.025, true)










