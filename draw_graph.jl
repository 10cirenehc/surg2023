include("graph_generation.jl")
using Plots

g = Graphs.loadgraph("data/n-custom_barabasi_albert-10-100-0.7.lg","g46.0")

draw(g)

g = Graphs.loadgraph("data/n-custom_erdos_renyi-10-50-0.7.lg","g46.0")
draw(g)

g = Graphs.loadgraph("data/n-custom_geometric-10-50-0.7.lg","g46.0")
draw(g)

chip_data = DataFrame(CSV.File("data/chip_data.txt", header=false))
scatter(chip_data[:,1],chip_data[:,2],zcolor=chip_data[:,3], legend = false, xlabel="x", ylabel="y", title="Cosmo Chip data")

# g = Graphs.loadgraph("data/n-custom_watts_strogatz-10-50-0.7.lg","g46.0")
# draw(g)