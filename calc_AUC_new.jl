# calculate AUC for one class
# needs code optimization
using PyCall
@pyimport sklearn.metrics as metrics

function calc_AUC_new(Rtest,Xa)
  Rtest = Rtest[:]
  Xa = Xa[:]
  minNonZero = minimum(abs(Xa[find(Xa)]))
  Xa = Xa / minNonZero
  @show length(Xa)
  @show length(find(Xa))
  fpr,tpr,thresholds = metrics.roc_curve(Rtest,Xa)
  auc = metrics.auc(fpr,tpr)
  return fpr,tpr,auc
end
