include("readSMAT.jl")
include("general_APTRANK.jl")
include("calc_AUC_new.jl")
include("splitRT.jl")
include("colnormout.jl")

mainloc = "data/yeast"
location = join([mainloc,"_H.smat"])
H = readSMAT(location)
lambda = 0.5
H1 = lambda*H;
H2 = (1-lambda)*H';
dH = colnormout(H1) + colnormout(H2);
location = join([mainloc,"_R.smat"])
R = readSMAT(location)

location = join([mainloc,"_G.smat"])
(rows,header) = readdlm(location;header=true)
G = sparse(
           convert(Array{Int64,1},rows[1:parse(Int,header[3]),1])+1,
           convert(Array{Int64,1},rows[1:parse(Int,header[3]),2])+1,
           rows[1:parse(Int,header[3]),3],
           parse(Int,header[1]),
           parse(Int,header[2])
           )
G = (G+G')/2
ratio = 0.5
Rtrain,Rtest = splitRT(R,ratio)
Gtrain = vcat(hcat(G,Rtrain),hcat(Rtrain',dH))
ei,ej,v = findnz(Gtrain)
G_pos = 1:size(G,1)
H_pos = (size(G,1)+1):(size(G,1)+size(H,1))
K = 8
S = 5
train_rows = (G_pos,)
train_cols = (H_pos,)
predict_rows = G_pos
predict_cols = H_pos
d = 2
Xa,alpha,all_alpha = general_APTRANK(ei,ej,v,Gtrain.m,Gtrain.n,train_rows,train_cols,
                                     predict_rows,predict_cols;diff_type=d,K=K,
                                     S=S,second_run = 0,ratio=ratio,method = "median")
fpr,tpr,auc = calc_AUC_new(Rtest,Xa)
