using Statistics
using CSV
using StatsBase
using DataFrames
using DataFramesMeta
using ArgParse
using Distributed
using ProgressBars
using MLJ
using MLJBase
using LightGBM
using StableRNGs
using CategoricalArrays

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--labels"
            help = "labels"
            action => :store_arg
            default = Any["E", "aE"]
            nargs='+'
        "--labelname"
            help = "label name"
            arg_type = Any
            default = "most_freq"
        "--attrfile"
            help = "attribute file"
            arg_type = Any
            required = true
    end
    return parse_args(s)
end

function impute_missing(df)
    nmissing = sum([count(ismissing,col) for col in eachcol(df)])
    printstyled("\tImputation of $nmissing missing values...\n", color = :cyan)
    numeric_columns = []
    for nm in names(df)
        if Float64 <: eltype(df[!, nm])
            append!(numeric_columns, [nm])
        end
    end                                   # select only numeric attributes
    df = DataFrames.select(df, numeric_columns)
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
    printstyled("\tNormalizing ...\n", color = :cyan)
    foreach(c -> c .= (c .- mean(c)) ./ std(c), eachcol(df)) 
end

function read_csv(filename) 
    printstyled("loading file ... $filename\n", color = :green)
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

    ttargets = parsed_args["labels"]
    attrfile = parsed_args["attrfile"]
    labelname = parsed_args["labelname"]

    df = read_csv(attrfile)
    df_attr = impute_missing(del_missing_col_rows(df))
    foreach(c -> c .= (c .- mean(c)) ./ std(c), eachcol(df_attr))  # z-score
    df_attr[!, :name] = df[!, :name]
    #df_labels = dropmissing(df[!, [:name, Symbol(labelname)]])   # drop genes with missing labels
	df_labels = read_csv("KIDNEY/node_labels.csv")
	df_labels = filter(Symbol(labelname) => !ismissing, df_labels)
	df_labels = filter(row -> row[Symbol(labelname)] ∈ ["E","NE"], df_labels)[!, Not(r"ACH-")]

    df_attr = innerjoin(df_attr,df_labels, on=:name)[!, Not([:name, Symbol(labelname)])]

    X = Matrix(df_attr)
    y = vec(Matrix(df_labels))
    y = CategoricalArray(y)
    rng = StableRNG(566)
    LIGHTGBM_SOURCE = abspath("~/LightGBM-3.2.0")
    model = LightGBM.MLJInterface.LGBMClassifier(
    objective = "multiclass", 
    num_iterations = 100,
    learning_rate = .1, 
    early_stopping_round = 15,
    num_leaves = 1000)
    global preds = []
    global targets = []
    stratified_cv = StratifiedCV(nfolds=5, rng=1234)
    rows = 1:size(X)[1]
    splits = MLJBase.train_test_pairs(stratified_cv, rows, y)
    printstyled("Classifying ...\n", color = :green)
    for s in ProgressBar(splits)
        train, test = s[1], s[2]
        mach = machine(model, X, y, scitype_check_level=0)
        MLJ.fit!(mach, rows=train, verbosity=-1) 
        targets = cat(targets, vec(y[test, :]), dims=1)
        preds = cat(preds, MLJ.predict_mode(mach, rows=test), dims=1)
    end
    ŷ = convert(CategoricalArray{String3,1,UInt32},  preds)
    y_targets = convert(CategoricalArray{String3,1,UInt32}, targets)
    println("Accuracy: \t$(MLJBase.Accuracy()(ŷ,y_targets))")
    println("Balanced Acc: \t$(MLJBase.BalancedAccuracy()(ŷ,y_targets))")
    println("MCC: \t\t$(MLJBase.MatthewsCorrelation()(ŷ,y_targets))")
    println(confusionmatrix(ŷ,y_targets))
end

main()

