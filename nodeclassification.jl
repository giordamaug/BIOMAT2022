using Statistics
using CSV
using StatsBase
using DataFrames
using DataFramesMeta
using WebIO
using ArgParse
using Distributed
using ProgressLogging
using MLJ

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--labels"
            help = "labels"
            action => :store_arg
            default = Any["E", "aE"]
            nargs='+'
        "--attrfile"
            help = "attribute file"
            arg_type = Any
            required = true
        "--labelfile"
            help = "label file"
            arg_type = Any
            required = true
    end
    return parse_args(s)
end

function impute_missing(df)
    nmissing = sum([count(ismissing,col) for col in eachcol(df)])
    println("Imputation of $nmissing missing values... ")
    numeric_columns = []
    for nm in names(df)
        if Float64 <: eltype(df[!, nm])
            append!(numeric_columns, [nm])
        end
    end                                   # select only numeric attributes
    df = select(df, numeric_columns)
    for col in eachcol(df)                # fix missing and naN with mean value
        m = mean(collect(skipmissing(col)))
        std = mean(collect(skipmissing(col)))
        replace!(col, missing => m)
        replace!(col, NaN => m)
        convert.(Float64,col)
    end
    return mapcols(ByRow(Float64), df)     # convert Union{missing,Float64} to Float64 types
end

function zscoring(df)
    dt = StatsBase.fit(ZScoreTransform, Matrix(df), dims=2)
    return StatsBase.transform(dt, Matrix(df))
end

function read_csv(filename) 
    println("loading file ...", filename)
    df = CSV.File("$filename", typemap=Dict(Int => Float64), missingstring=["NaN", "NA"]) |> DataFrame       # read the attribute file
    df = unique(df, :name)                  # remove possible duplicated genes
    return df
end

function del_missing_col_rows(df)
    df = filter(x -> any(!ismissing, x[Not(r"name|label")]), df)  # remove rows with all missing (except in pattern)
    return df[!, map(x->!all(ismissing, x), eachcol(df))]
end

function confusionmatrix(predictions, labels)
    classes = vec(unique(labels))
    d = size(classes)[1]
    idx = Dict(zip(classes,Vector(1:d)))
    c = zeros(Int64, d,d)
    for i in 1:size(labels)[1]
       c[idx[labels[i]] ,idx[predictions[i]]] += 1
    end
    df = DataFrame(c, classes)
    df[!, :name] = classes
    return df
end

function main()
    parsed_args = parse_commandline()

    targets = parsed_args["labels"]
    attrfile = parsed_args["attrfile"]
    labelfile = parsed_args["labelfile"]
    
    df = read_csv(attrfile)
    df_attr = impute_missing(del_missing_col_rows(df))
    df_attr[!, :name] = df[!, :name]
    df_labels = dropmissing(df[!, [:name, :label_wo_outliers]])   # drop genes with missing labels
    df_attr = innerjoin(df_attr,df_labels, on=:name)[!, Not([:name, :label_wo_outliers])]

    X = Matrix(df_attr)
    y = vec(Matrix(df_labels))

end

main()

