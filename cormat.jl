#
# run cmd+alt+p on MacOS in SublimeText and enter Build-with: julia-build(Or from Tool menu )
# 
#
using Statistics
using CSV
using DataFrames

df = CSV.File("./KIDNEY/attrib.csv", drop=[1]) |> DataFrame
cormat = cor(Array(df))
cormat[cormat.<0.1] .= 0
#using LightGraphs
using Graphs, GraphPlot
using Compose
import Cairo, Fontconfig

nodelabels=names(df)
G = SimpleGraph(cormat)
layout=(args...)->spring_layout(args...; C=200)
draw(PNG("cormat.png", 16cm, 16cm), gplot(G,nodelabel=nodelabels))

