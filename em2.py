import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import logsumexp

datadir = "DATA"
tissue = "lung"
df = pd.read_csv(f"{datadir}/CRISPR_{tissue}_scores_wo_outliers.tsv", sep='\t', index_col=0)
#x_unlabeled = data_unlabeled[["ACH.000159", "ACH.000189"]].values
print(f'Matrix has {len(df)} rows.')
naninrows = sum([True for idx,row in df.iterrows() if any(row.isnull())])
print(f'There are {naninrows} rows with Nan... removing them!')
#df = df.dropna()
#print(f'Matrix has now {len(df)} rows.')
if naninrows > 0:
    fill_mean = lambda row : row.fillna(row.mean())
    df = df.apply(fill_mean, axis = 1)

x_unlabeled = df.values

def get_random_psd(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())


def initialize_random_params(size):
    params = {'phi': np.random.uniform(0, 1),
              'mu0': np.random.normal(0, 1, size=(size,)),
              'mu1': np.random.normal(0, 1, size=(size,)),
              'sigma0': get_random_psd(size),
              'sigma1': get_random_psd(size)}
    return params

def e_step(x, params):
    np.log([stats.multivariate_normal(params["mu0"], params["sigma0"]).pdf(x),
            stats.multivariate_normal(params["mu1"], params["sigma1"]).pdf(x)])
    log_p_y_x = np.log([1-params["phi"], params["phi"]])[np.newaxis, ...] + \
                np.log([stats.multivariate_normal(params["mu0"], params["sigma0"]).pdf(x),
            stats.multivariate_normal(params["mu1"], params["sigma1"]).pdf(x)]).T
    log_p_y_x_norm = logsumexp(log_p_y_x, axis=1)
    return log_p_y_x_norm, np.exp(log_p_y_x - log_p_y_x_norm[..., np.newaxis])

def m_step(x, params):
    total_count = x.shape[0]
    _, heuristics = e_step(x, params)
    heuristic0 = heuristics[:, 0]
    heuristic1 = heuristics[:, 1]
    sum_heuristic1 = np.sum(heuristic1)
    sum_heuristic0 = np.sum(heuristic0)
    phi = (sum_heuristic1/total_count)
    mu0 = (heuristic0[..., np.newaxis].T.dot(x)/sum_heuristic0).flatten()
    mu1 = (heuristic1[..., np.newaxis].T.dot(x)/sum_heuristic1).flatten()
    diff0 = x - mu0
    sigma0 = diff0.T.dot(diff0 * heuristic0[..., np.newaxis]) / sum_heuristic0
    diff1 = x - mu1
    sigma1 = diff1.T.dot(diff1 * heuristic1[..., np.newaxis]) / sum_heuristic1
    params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
    return params

def get_avg_log_likelihood(x, params):
    loglikelihood, _ = e_step(x, params)
    return np.mean(loglikelihood)


def run_em(x, params):
    avg_loglikelihoods = []
    while True:
        avg_loglikelihood = get_avg_log_likelihood(x, params)
        avg_loglikelihoods.append(avg_loglikelihood)
        if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
            break
        params = m_step(x, params)
    print("\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s" % (params['phi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']))
    _, posterior = e_step(x, params)
    forecasts = np.argmax(posterior, axis=1)
    return forecasts, posterior, avg_loglikelihoods

random_params = initialize_random_params(len(df.columns))
unsupervised_forecastsforecasts, unsupervised_posterior, unsupervised_loglikelihoods = run_em(x_unlabeled, random_params)
print("total steps: ", len(unsupervised_loglikelihoods))
plt.plot(unsupervised_loglikelihoods)
plt.title("unsupervised log likelihoods")
plt.savefig(f"{tissue.capitalize()}_unsupervised.png")
plt.close()

values, counts = np.unique(unsupervised_forecastsforecasts, return_counts=True)
minorityindex = np.argmin(counts)
minoritylabel = values[minorityindex]
majorityindex = np.argmax(counts)
majoritylabel = values[majorityindex]
alias = {minoritylabel : 'E+aE', majorityindex: 'NE'}
print(values,counts, minorityindex, minoritylabel)
dfseries = pd.Series(unsupervised_forecastsforecasts, index=df.index)
dfseries = dfseries.replace(0, alias[0])
dfseries = dfseries.replace(1, alias[1])
print(dfseries)
df["label_EM"] = dfseries
#df.to_csv("DATA/KidneyLabellingEM_DaScores.csv", index=True)

df_E = df[df["label_EM"] == 'E+aE']
df_E = df_E.drop(columns=['label_EM'])
xe_unlabeled = df_E.values
random_params_E = initialize_random_params(len(df_E.columns))
unsupervised_forecastsforecasts_E, unsupervised_posterior_E, unsupervised_loglikelihoods_E = run_em(xe_unlabeled, random_params_E)
print(unsupervised_forecastsforecasts_E.shape, xe_unlabeled.shape)
values_E, counts_E = np.unique(unsupervised_forecastsforecasts_E, return_counts=True)
print(values_E,counts_E)
minorityindex = np.argmin(counts_E)
minoritylabel = values[minorityindex]
majorityindex = np.argmax(counts_E)
majoritylabel = values[majorityindex]
alias_E = {minoritylabel : 'aE', majorityindex: 'E'}
dfseries_E = pd.Series(unsupervised_forecastsforecasts_E, index=df_E.index)
dfseries_E =  dfseries_E.replace(0, alias_E[0])
dfseries_E = dfseries_E.replace(1, alias_E[1])
df_E["label_EM"] = dfseries_E
print("total steps: ", len(unsupervised_loglikelihoods_E))
plt.plot(unsupervised_loglikelihoods_E)
plt.title("unsupervised log likelihoods E")
plt.savefig(f"{tissue.capitalize()}_unsupervised_E.png")
plt.close()

#df_E.to_csv("DATA/KidneyLabellingEM_DaScores_E.csv", index=True)
pd.concat([df[df["label_EM"] == 'NE'], df_E]).to_csv(f"{datadir}/{tissue.capitalize()}LabellingEM_DaScores_3class.csv", index=True)
