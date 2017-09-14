
include("include_all.jl")
include("get_data_yeast.jl")

Xa,alpha,all_alpha = general_APTRANK(ei,ej,v,Gtrain.m,Gtrain.n,train_rows,train_cols,
                                     predict_rows,predict_cols;diff_type=d,K=K,
                                     S=S,second_run = 0,ratio=ratio,method = "median")
fpr,tpr,auc = calc_AUC_new(Rtest,Xa)
