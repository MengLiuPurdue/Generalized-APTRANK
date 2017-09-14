function read_data(filename)

include(joinpath(Pkg.dir(),"handy_julia","handy_julia.jl"))

M,H = readdlm(filename,'\t',header=true)

all_gcd = vcat(M[:,1],M[:,3])
length(unique(all_gcd))

# split_at_firstchar = map(x->split(x,'.'),M[:,1])
# correctM2 = map(x->split_at_firstchar[x][1],1:length(split_at_firstchar))
#
# split_at_firstchar = map(x->split(x,'.'),M[:,3])
# correctM4 = map(x->split_at_firstchar[x][1],1:length(split_at_firstchar))

# genes_ids_ei = find(correctM2.=="G")
# genes_ids_ej = find(correctM4.=="G")

# only_genes = vcat(M[genes_ids_ei,1],M[genes_ids_ej,3])
# ugenes = unique(only_genes)

genes_ids_ei = find(M[:,2].=="Gene")
genes_ids_ej = find(M[:,4].=="Gene")
only_genes = vcat(M[genes_ids_ei,1],M[genes_ids_ej,3])
ugenes = String.(unique(only_genes))

# uonly_genes = unique(vcat(M[genes_ids_ei,1],M[genes_ids_ej,3]))
# sp = map(x->split(x,'.'),uonly_genes)
# fl = map(x->sp[x][1],1:length(sp))
# only_genes_fromto = 1:length(only_genes)
# for i = 1:length(fl)
#   if length(sp[i]) == 2
#     if fl[i] != "G"
#       println("i is $i")
#     end
#   end
# end
# length(unique(only_genes))

chemical_ids_ei = find(M[:,2].=="Chemical")
chemical_ids_ej = find(M[:,4].=="Chemical")
only_chemicals = vcat(M[chemical_ids_ei,1],M[chemical_ids_ej,3])
uchemicals = String.(unique(only_chemicals))

disease_ids_ei = find(M[:,2].=="Disease")
disease_ids_ej = find(M[:,4].=="Disease")
only_diseases = vcat(M[disease_ids_ei,1],M[disease_ids_ej,3])
udiseases = String.(unique(only_diseases))

EI = zeros(Int64,size(M,1))
EJ = zeros(Int64,size(M,1))

gene_size = length(ugenes)
chem_size = length(uchemicals)
dis_size = length(udiseases)
G_size = gene_size+chem_size+dis_size

# @show typeof(vec(String.(ugenes)))
# @show typeof(vec(String.(M[genes_ids_ei,1])))
# M = String.(M)
EI[genes_ids_ei] = findin_index(String.(M[genes_ids_ei,1]),ugenes)
EJ[genes_ids_ej] = findin_index(String.(M[genes_ids_ej,3]),ugenes)

EI[chemical_ids_ei] = findin_index(String.(M[chemical_ids_ei,1]),uchemicals) + gene_size
EJ[chemical_ids_ej] = findin_index(String.(M[chemical_ids_ej,3]),uchemicals) + gene_size

EI[disease_ids_ei] = findin_index(String.(M[disease_ids_ei,1]),udiseases) + gene_size + chem_size
EJ[disease_ids_ej] = findin_index(String.(M[disease_ids_ej,3]),udiseases) + gene_size + chem_size

G = sparse(EI,EJ,1,G_size,G_size)

return G,gene_size,chem_size,dis_size,ugenes,uchemicals,udiseases
end
