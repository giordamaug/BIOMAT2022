import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import logsumexp

df = pd.read_csv("DATA/CRISPR_kidney_scores_wo_outliers.tsv", sep='\t', index_col=0)
print(df)
#x_unlabeled = data_unlabeled[["ACH.000159", "ACH.000189"]].values
print(f'Matrix has {len(df)} rows.')
print(f'There are {sum([True for idx,row in df.iterrows() if any(row.isnull())])} rows with Nan... removing them!')
df = df.dropna()
print(f'Matrix has now {len(df)} rows.')
xdf = df.copy()
xdf['mean'] = df.mean(axis=1)
xdf = xdf.sort_values(['mean'])
df_labeled = pd.concat([df.loc[xdf.head(300).index], df.loc[xdf.tail(1300).index]])
df_labeled["label"] = np.array([0]*300 + [1]*1300)
df_unlabeled = xdf.iloc[300:-1300]
df_unlabeled["label"] = pd.Series([0 for _ in range(len(df_unlabeled.index))])
x_unlabeled = df.loc[df_unlabeled.index].values
x_labeled = df.loc[df_labeled.index].values
y_labeled = np.array([0]*300 + [1]*1300)

def get_random_psd(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())


def learn_params(x_labeled, y_labeled):
    n = x_labeled.shape[0]
    phi = x_labeled[y_labeled == 1].shape[0] / n
    mu0 = np.sum(x_labeled[y_labeled == 0], axis=0) / x_labeled[y_labeled == 0].shape[0]
    mu1 = np.sum(x_labeled[y_labeled == 1], axis=0) / x_labeled[y_labeled == 1].shape[0]
    sigma0 = np.cov(x_labeled[y_labeled == 0].T, bias= True)
    sigma1 = np.cov(x_labeled[y_labeled == 1].T, bias=True)
    return {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}

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
        params = m_step(x_unlabeled, params)
    print("\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
               % (params['phi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']))
    _, posterior = e_step(x_unlabeled, params)
    forecasts = np.argmax(posterior, axis=1)
    return forecasts, posterior, avg_loglikelihoods

learned_params = learn_params(x_labeled, y_labeled)
semisupervised_forecasts, semisupervised_posterior, semisupervised_loglikelihoods = run_em(x_unlabeled, learned_params)
print("total steps: ", len(semisupervised_loglikelihoods))
plt.plot(semisupervised_loglikelihoods)
plt.title("semi-supervised log likelihoods")
plt.savefig("semi-supervised.png")

values, counts = np.unique(semisupervised_forecasts, return_counts=True)
print(values,counts)
df_unlabeled["label"] = semisupervised_forecasts
print(df_labeled)
print(df_unlabeled)
pd.concat([df_labeled["label"], df_unlabeled["label"]]).to_csv("DATA/KidneyLabellingSEM_DaScores.csv", index=True)