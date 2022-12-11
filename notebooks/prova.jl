using CSV
using DataFrames
using GeometricFlux
using Node2Vec
using GraphSignals
using Graphs

df = CSV.File("/Users/maurizio/BIOMAT2022/KIDNEY/ppi.simple.csv") |> DataFrame

genes = union(Set(df[!, :source]),Set(df[!, :target]))
idx = Dict(genes .=> 1:length(genes))
edgelista = (x,y) -> (idx[x],idx[y])
transform!(df, [:source, :target] => ByRow((x,y) -> edgelista(x, y)) => :edge)
display(df)
#g = Graphs.SimpleGraph(Edge.(df[!, :edge]))
#walks = simulate_walks(g,10,80,1.0,1.0)
#model = learn_embeddings(walks)
#vectors = model.vectors
g = Graphs.SimpleGraphs.SimpleGraph(length(genes))
for i in 1:size(df)[1]
    Graphs.SimpleGraphs.add_edge!(g, df[i, :edge][1], df[i, :edge][2])
end
fg = FeaturedGraph(g)
vectors = node2vec(fg; walks_per_node=10, len=80, p=1.0, q=1.0)