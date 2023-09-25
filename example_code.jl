using LinearAlgebra, Plots, Convex, Mosek, MosekTools, JuMP, LaTeXStrings, Printf, Measures, Serialization, Graphs

include("func_library.jl")
include("graph_generation.jl")

# This code provides an introduction to using the function library
# The result of this code is the classification example at the 
# end of Donato Ridgley et al 2021 without the SVL results


iter = 1000


# Make Graph
# This is where you should plug in code to generate random graphs

# Start A from CDC2021 paper

#g = custom_erdos_renyi(20,8)
A = 0.25*[0 1 0 1 0 1 0;
    0 0 1 0 1 0 1;
    1 0 0 1 0 1 0;
   0 1 0 0 1 0 1;
    1 0 1 0 0 1 0;
    0 1 0 1 0 0 1;
    1 0 1 0 1 0 0]

#A = 0.1.*adjacency_matrix(g)

n,n = size(A)
L = Diagonal(A*ones(n)) - A

println(ones(1,n)*L)

# Make objective functions
# Note, the output logistic_regression_COSMO_chip_data results in a random partitioning of the data
# therefore, the output will be slightly different everytime it is called
# To make the comparision fair, save the result using Serialize or by 
# putting the code into jupyter notebook

data,X,Y,H,Gam = logistic_regression_COSMO_chip_data(n,6,0,1)

X = X'
d = size(X)[2]

f = x -> LogReg(x,H,Gam)

# Calculate sector bounds
sigma = opnorm(I-1/n*ones(n,n)-L)
@printf("sigma = %0.5f\n", sigma)

m = 2/n
Lip = maximum([estL(H[i],n) for i = 1:n])
kappa = Lip/m


# Packet Loss Probability
pl = 0.3

# Optimize SH-SVL parameters
# This solves the LMI from the paper repeatedly and finds alg parameters
# that minimize the worst-case convergence rate rho for given kappa and sigma
# You will use this value of rho in your analysis

# You may have to preclude results with rho=1 since they do not converge in 
# the wost case and the "optimal" parameters will be junk

# Note that the optimization of the alg parameters can be slow. The speed is
# greatly increased if the search for the optimum is warmstarted to a nearby point
# this can be achieved by setting param_init to be optparam from the last run if 
# kappa and sigma do not vary greatly

localenv = [kappa, sigma]#, 1-pl] kappa = L/m
tol = 1e-5
global k = 0
global param_init = Diagonal([1,2,2,2])*rand(4) # warmstarting will speed the following code up
max_iter = 4000
while k < max_iter && find_rho(test_rho,param_init,localenv)==1.0
    global param_init = Diagonal([1,2,2,2])*rand(4)
    global k += 1
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
alpha, delta, eta, zeta = optparam

@printf("L = %0.4f, and kappa = %0.4f\n", Lip, kappa)

# Compiles the state matrices using the optimal parameters
# Must weight the step size by the Lipschitz constant since 
# LMI was solved wrt kappa

G = [1 0 -alpha/Lip -zeta;
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




# compute maximum error
# Here is where you will need to compute the empirical convergence rate
# There will be an initial transient before the error begins to decay exponentially
# on a log scale the decay will appear linear and the rate is the slope on the semilog axis
# you will need to write some simple code to determine when the transient is over and then compute
# the convergence rate

e = [maximum([norm(x[i,:,k].-xopt,Inf) for i=1:n]) for k=1:lossless_time]
e2 = [maximum([norm(xp[i,:,k].-xopt,Inf) for i=1:n]) for k=1:lossy_time]

# Here I computed the rate in a very naive fashion by assuming that the transient will end within 
# 100 steps and that no numerical issues will be encountered by 200 steps
# you will likely need additional guard rails before gathering data to ensure nothing 
# unexpected happens

r = (e[200]/e[100])^(1/100)
r2 = (e2[200]/e2[100])^(1/100)

@printf("Alg 2 lossless rho = %0.4f\n", r)
@printf("Alg 2 lossy rho = %0.4f\n", r2)

@printf("L = %0.4f, and kappa = %0.4f\n", Lip, kappa)



# Plot
err = L"\max_i ||x_i-x_{opt}||_\infty" # using LaTeXStrings to embed math in the label
plot(e, yaxis=:log,ylabel=err,xlabel="iterations",label="Alg 2, lossless",color=1,linestyle=:solid, size=[800,600], margin=10Measures.mm)
plot!(e2, yaxis=:log,ylabel=err,xlabel="iterations",label="Alg 2, lossy",color=1,linestyle=:dash)

savefig("traces_24.png")