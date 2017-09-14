# example code to run Generalized Aptrank
mainloc = "data/yeast"
location = join([mainloc,"_H.smat"])
H = readSMAT(location)
location = join([mainloc,"_R.smat"])
R = readSMAT(location)
location = join([mainloc,"_G.smat"])
G = readSMAT_FLOAT(location) # G = Int64.(spones(G)) # if you want the 1/0 version
G = (G+G')/2


lambda = 0.5
H1 = lambda*H;
H2 = (1-lambda)*H';
dH = colnormout(H1) + colnormout(H2);
dH = (dH+dH')/2

ratio = 0.5
Rtrain,Rtest = splitRT(R,ratio)

Gtrain = vcat(hcat(G,Rtrain),hcat(Rtrain',dH))
ei,ej,v = findnz(Gtrain)
G_pos = 1:size(G,1)
H_pos = (size(G,1)+1):(size(G,1)+size(H,1))
K = 7
S = 3
train_rows = (G_pos,)
train_cols = (H_pos,)
predict_rows = G_pos
predict_cols = H_pos
d = 2
