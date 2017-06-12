# Generalized APTRANK
# Input:
# ei, ej       - an edge list
# m, n         - the number of rows and edges of the original data set (to ensure
#                the reconstructed matrix has the same dimension of the original one)
# train_rows   - the row positions of training data, represented in a tuple of arrays
# train_cols   - the column positions of training data, represented in a tuple of arrays
# predict_rows - the row positions where you want predictions
# predict_cols - the column positions where you want predictions
# K            - the number of terms used in the APTRANK, default is 8
# S            - the number of independent experiments during training,
#               default is 5
# diff_type    - choose what kind of diffusion matrix you want, 1 for G/rho, 2 for
#               G*D^{-1}, 3 for (D - G)/rho and 4 for (I - D^{-1/2}*G*D^{-1/2}),
#               where rho is the spectral radius and D is the out-degree of each
#               vertex, defalut is 1
# ratio        - the split ratio between fiiting set and validation set, default is 0.8
# rho          - a parameter used to control the diffusion process, rho > 0, if a
#                rho <= 0 is given, it will be replaced by the spectral radius
# seeds        - seed positions, where the diffusion starts, default is the same as predict_cols
# sampling_type- 1 for randomly sampling using ratio, 2 for S-fold cross validation, default is 1
#
#
# Output:
# Xa            - the predicted matrix, zero rows or columns in the original graph
#                 will remain zero.
# alpha         - the final parameters of APTRANK
# all_alpha     - records of alphas in each independent experiment

using Convex
using ECOS
using StatsBase
using MLBase
include("splitRT.jl")
include("manage_procs.jl")

function general_APTRANK(ei,ej,m,n,train_rows,train_cols,predict_rows,predict_cols;
                         K = 8,S = 5,diff_type = 1,ratio = 0.8,rho = 0.0,seeds = predict_cols,
                         sampling_type = 1)

  nrows = length(predict_rows)
  ncols = length(predict_cols)
  G = sparse(ei,ej,1,m,n)
  G = round(Int64,G)
  if !issymmetric(G)
    error("The input must be symmetric.")
  end
  print("symmetric check success!\n")
  np = nprocs()
  if np < 12
    addprocs(12-np)
  end
  np = nprocs()
  all_alpha = zeros(Float64,K-1,S)
  if sampling_type == 2
    print("start sampling S-fold cross validation\n")
    folds = Array{Vector{Any}}(length(train_rows))
    for i = 1:length(train_rows)
      nr = length(train_rows[i])
      nc = length(train_cols[i])
      @time folds[i] = collect(Kfold(nr*nc,S))
    end
  end
  for s = 1:S
    b = []
    positions = Array(Array{Int64},length(train_rows))
    Gf = copy(G)
    if sampling_type == 1
      for i = 1:length(train_rows)
        nr = length(train_rows[i])
        nc = length(train_cols[i])
        set = round(Int64,linspace(1,nr*nc,nr*nc))
        pf = sample(set,round(Int64,ratio*nr*nc),replace = false)
        pv = setdiff(set,pf)
        training = G[train_rows[i],train_cols[i]] + 1
        ii,jj,vv = findnz(training)
        Rf = sparse(ii[pf],jj[pf],vv[pf]-1,nr,nc)
        Gf[train_rows[i],train_cols[i]] = Rf
        b = vcat(b,vv[pv]-1)
        positions[i] = pv
      end
    elseif sampling_type == 2
      print("extract a new fold.\n")
      for i = 1:length(train_rows)
        nr = length(train_rows[i])
        nc = length(train_cols[i])
        set = round(Int64,linspace(1,nr*nc,nr*nc))
        pf = set[folds[i][s]]
        pv = setdiff(set,pf)
        training = G[train_rows[i],train_cols[i]] + 1
        ii,jj,vv = findnz(training)
        Rf = sparse(ii[pf],jj[pf],vv[pf]-1,nr,nc)
        Gf[train_rows[i],train_cols[i]] = Rf
        b = vcat(b,vv[pv]-1)
        positions[i] = pv
      end
    else
      error("Invalid sampling type!")
    end
    @show s
    F = get_diff_matrix(Gf,diff_type,rho)
    @eval @everywhere F = $F
    Gf = 0
    gc()
    X0 = spzeros(G.m,G.n)
    for i = 1:length(seeds)
      X0[seeds[i],seeds[i]] = 1
    end
    N = length(seeds)
    bs = ceil(Int64,N/np)
    nblocks = ceil(Int64,N/bs)

    all_ranges = Array(Array{Int64},nblocks)
    for i = 1:nblocks
      start = 1 + (i-1)*bs
      all_ranges[i] = seeds[start:min(start+bs-1,N)]
    end
    for i = 1:np
      t =  X0[:,all_ranges[i]]
      sendto(i,X = t)
    end
    X0 = 0
    gc()
    Arows = zeros(Int64,length(train_rows)+1)
    for i = 2:(length(train_rows)+1)
      Arows[i] = Arows[i - 1] + length(positions[i - 1])
    end
    A = zeros(Float64,Arows[length(Arows)],K-1)
    #@show "start"
    for k = 1:K
      @show k
      #@show size(X),size(F)
      @time @everywhere X = F * X
      k == 1 && continue
      Xh = zeros(G.m,G.n)
      for i = 1:np
        Xi = getfrom(i,:X)
        #@show size(Xh[:,all_ranges[i]])
        #@show size(Xi[predict_rows,:])
        @time Xh[:,all_ranges[i]] = Xi
        Xi = 0
        gc()
      end
      for i = 1:length(train_rows)
        validating = Xh[train_rows[i],train_cols[i]]
        validating = validating[:]
        A[(Arows[i]+1):(Arows[i+1]),k-1] = validating[positions[i]]
      end
      Xh = 0
      gc()
    end
    #@show size(A)
    #@show sum(isnan(A))
    #@show findnz(A)
    @eval @everywhere X,F = 0,0
    @everywhere gc()
    Qa,Ra = qr(A)
    A = 0
    gc()
    alpha = Variable(size(Ra,2))
    print("start solving Least Sqaure\n")
    @show Ra
    #@show findnz(Ra)
    #@show findnz(Qa'*b)
    problem = minimize(norm(Qa'*b - Ra*alpha),alpha >= 0, sum(alpha) == 1)
    #solve!(problem, GurobiSolver())
    solve!(problem,ECOSSolver())
    Qa = 0
    Ra = 0
    b = 0
    gc()
    all_alpha[:,s] = alpha.value
    @show alpha.value
  end
  alpha = mean(all_alpha,2)
  alpha = alpha / sum(alpha)
  F = get_diff_matrix(G,diff_type,rho)
  @eval @everywhere F = $F
  X0 = zeros(G.m,G.n)
  for i = 1:length(seeds)
    X0[seeds[i],seeds[i]] = 1
  end
  N = length(seeds)
  bs = ceil(Int64,N/np)
  nblocks = ceil(Int64,N/bs)

  all_ranges = Array(Array{Int64},nblocks)
  for i = 1:nblocks
    start = 1 + (i-1)*bs
    all_ranges[i] = seeds[start:min(start+bs-1,N)]
  end
  for i = 1:np
    t =  X0[:,all_ranges[i]]
    sendto(i,X = t)
  end
  X0 = 0
  gc()
  A = zeros(Float64,nrows*ncols,K-1)
  #@show "start"
  for k = 1:K
    @show k
    #@show size(X),size(F)
    @time @everywhere X = F * X
    k == 1 && continue
    Xh = zeros(G.m,G.n)
    for i = 1:np
      Xi = getfrom(i,:X)
      #@show size(Xh[:,all_ranges[i]])
      #@show size(Xi[predict_rows,:])
      @time Xh[:,all_ranges[i]] = Xi
      Xi = 0
      gc()
    end
    ii,jj,vv = findnz(Xh[predict_rows,predict_cols])
    @show sum(isnan(vv))
    rowids = ii + (jj - 1)*(nrows)
    A[rowids,k-1] = vv
    Xh = 0
    gc()
  end
  Xa = A * alpha
  #@show "reshape"
  Xa = reshape(Xa,nrows,ncols)
  @show alpha

  return Xa,alpha,all_alpha
end

function get_diff_matrix(G,diff_type,rho)
  #@show issymmetric(G)
  d = vec(sum(G,2))
  #@show maximum(d),minimum(d)
  dinv = 1./d
  for i = 1:length(dinv)
    if dinv[i] == Inf
      dinv[i] = 0
    end
  end
  D = sparse(diagm(d))
  Dinv = sparse(diagm(dinv))
  Droot = sparse(diagm(sqrt(dinv)))
  L = G - D
  #@show size(G)
  #@show "done"
  if diff_type == 1
    if rho <= 0
      rho = maximum(abs(eigs(sparse(G),which = :LM, ritzvec = false)[1]))
      @show rho
    end
    F = G / rho
  elseif diff_type == 2
    F = G * Dinv
  elseif diff_type == 3
    if rho <= 0
      rho = maximum(abs(eigs(sparse(L),which = :LM, ritzvec = false)[1]))
      @show rho
    end
    F = L / rho
  elseif diff_type == 4
    F = -0.5*(speye(size(G,1)) - Droot * G * Droot)
  else
    error("unknown diffusion type")
  end
  #@show "return"
  return F
end
