function readSMAT(filename::AbstractString)
  (rows,header) = readdlm(filename;header=true)
  A = sparse(
             convert(Array{Int64,1},rows[1:parse(Int,header[3]),1])+1,
             convert(Array{Int64,1},rows[1:parse(Int,header[3]),2])+1,
             ones(Int64,parse(Int,header[3])),
             parse(Int,header[1]),
             parse(Int,header[2])
             )
  return A
end
