using Statistics
using CSV
using DataFrames
using DataFramesMeta
using PlotlyJS
using WebIO
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--labels"
            help = "labels"
            action => :store_arg
            default = Any["E", "aE"]
            nargs='+'
    end
    return parse_args(s)
end

function main()
    @show parsed_args = parse_commandline()

    targets = parsed_args["labels"]
    print(targets)
    
    df = CSV.File("./KIDNEY/node_attributes.csv") |> DataFrame
    df_labels = CSV.File("./KIDNEY/node_labels.csv") |> DataFrame
    labs = @rsubset(df_labels[!,(["name", "most_freq"])], :most_freq in targets)
    reduced_df = innerjoin(df,labs, on=:name)
    println("Working on ", nrow(reduced_df), " genes")
    x = reduced_df[!, Not([:name, :most_freq])] # exclude name and label column
    mapcols!(x) do y
             replace(y, missing => mean(skipmissing(y)))
    end

    cormat = cor(Matrix(x))   
    display(plot(heatmap(z=cormat, x=names(x), y=names(x))))
    readline()
end

main()

