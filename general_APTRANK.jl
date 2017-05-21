# Generalized APTRANK
# Input:
# ei, ej       - an edge list
# train_rows   - the row positions of training data
# train_cols   - the column positions of training data
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
#
# Output:
# Xa            - the predicted matrix, zero rows or columns in the original graph
#                 will remain zero.

using Convex
using SCS
using StatsBase
include("splitRT.jl")
include("manage_procs.jl")

function general_APTRANK(ei,ej,train_rows,train_cols,predict_rows,predict_cols;
                         K = 8,S = 5,diff_type = 1,ratio = 0.8,rho = 0.0)

  nrows = length(predict_rows)
  ncols = length(predict_cols)
  G = sparse(ei,ej,1)
  G = round(Int64,G)
  np = nprocs()
  if np < 12
    addprocs(12-np)
  end
  np = nprocs()
  all_alpha = zeros(Float64,K,S)
  for s = 1:S
    @show s
    Gf,Gv = splitRT(G,ratio)
    Rf = Gf[train_rows,train_cols]
    Rv = Gv[predict_rows,predict_cols]
    Gf,Gv = 0,0
    gc()
    newG = G
    newG[train_rows,train_cols] = Rf
    newG[train_cols,train_rows] = Rf'
    Rf = 0
    gc()
    F = get_diff_matrix(newG,diff_type,rho)
    @eval @everywhere F = $F
    newG = 0
    gc()
    X0 = spzeros(G.m,ncols)
    for i = 1:ncols
      X0[predict_cols[i],i] = 1
    end
    N = size(X0,2)
    bs = ceil(Int64,N/np)
    nblocks = ceil(Int64,N/bs)

    all_ranges = Array(UnitRange{Int64},nblocks)
    for i = 1:nblocks
      start = 1 + (i-1)*bs
      all_ranges[i] = start:min(start+bs-1,N)
    end
    for i = 1:np
      t =  X0[:,all_ranges[i]]
      sendto(i,X = t)
    end
    A = zeros(Float64,nrows*ncols,K)
    #@show "start"
    for k = 1:K
      @show k
      #@show size(X),size(F)
      @time @everywhere X = F * X
      #k == 1 && continue
      Xh = spzeros(Rv.m,Rv.n)
      for i = 1:np
        Xi = getfrom(i,:X)
        #@show size(Xh[:,all_ranges[i]])
        #@show size(Xi[predict_rows,:])
        @time Xh[:,all_ranges[i]] = Xi[predict_rows,:]
      end
      ii,jj,vv = findnz(Xh)
      #@show sum(isnan(vv))
      rowids = ii + (jj - 1)*(Xh.m)
      A[rowids,k] = vv
      Xh = 0
      gc()
    end
    #@show size(A)
    #@show sum(isnan(A))
    #@show findnz(A)
    Qa,Ra = qr(A)
    A = 0
    gc()
    b = reshape(Rv,prod(size(Rv)),1)
    alpha = Variable(size(Ra,2))
    print("start solving Least Sqaure\n")
    #@show Ra
    #@show findnz(Ra)
    #@show findnz(Qa'*b)
    problem = minimize(norm(Qa'*b - Ra*alpha),alpha >= 0, sum(alpha) == 1)
    #solve!(problem, GurobiSolver())
    solve!(problem)
    all_alpha[:,s] = alpha.value
  end
  alpha = mean(all_alpha,2)
  alpha = alpha / sum(alpha)
  F = get_diff_matrix(G,diff_type,rho)
  @eval @everywhere F = $F
  X0 = spzeros(G.m,ncols)
  for i = 1:ncols
    X0[predict_cols[i],i] = 1
  end
  N = size(X0,2)
  bs = ceil(Int64,N/np)
  nblocks = ceil(Int64,N/bs)
  all_ranges = Array(UnitRange{Int64},nblocks)
  for i = 1:nblocks
    start = 1 + (i-1)*bs
    all_ranges[i] = start:min(start+bs-1,N)
  end
  for i = 1:np
    t =  X0[:,all_ranges[i]]
    sendto(i,X = t)
  end
  A = zeros(Float64,nrows*ncols,K)
  #@show "start"
  for k = 1:K
    @show k
    #@show size(X),size(F)
    @time @everywhere X = F * X
    #k == 1 && continue
    Xh = spzeros(nrows,ncols)
    for i = 1:np
      Xi = getfrom(i,:X)
      @time Xh[:,all_ranges[i]] = Xi[predict_rows,:]
    end
    ii,jj,vv = findnz(Xh)
    #@show sum(isnan(vv))
    rowids = ii + (jj - 1)*(nrows)
    A[rowids,k] = vv
    Xh = 0
    gc()
  end
  Xa = A * alpha
  #@show "reshape"
  Xa = reshape(Xa,nrows,ncols)

  return Xa
end

function get_diff_matrix(G,diff_type,rho)
  #@show issymmetric(G)
  d = vec(sum(G,1))
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
  L = D - G
  #@show size(G)
  #@show "done"
  if diff_type == 1
    if rho <= 0
      rho = maximum(abs(eigs(sparse(G),which = :LM, ritzvec = false)[1]))
    end
    F = G / rho
  elseif diff_type == 2
    F = G * Dinv
  elseif diff_type == 3
    if rho <= 0
      rho = maximum(abs(eigs(sparse(L),which = :LM, ritzvec = false)[1]))
    end
    F = L / rho
  elseif diff_type == 4
    F = Droot * L * Droot
  else
    error("unknown diffusion type")
  end
  #@show "return"
  return F
end
