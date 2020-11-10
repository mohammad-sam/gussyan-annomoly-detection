
from numpy import exp, pi, dot, sqrt
from numpy.linalg import inv, det
from matplotlib.pyplot import scatter, show, hist


def gaussian_anomaly_detection(data):
    rows, cols = data.shape
    mu = data.mean(axis=0)
    diff = data - mu
    cov = dot(diff.T, diff) / rows
    a = exp(-0.5 * dot(dot(diff, inv(cov)), diff.T))
    b = sqrt(pow(2 * pi, cols) * det(cov))
    res = (a / b).sum(axis=1)
    return res


from sklearn.datasets import load_wine as load
from sklearn.decomposition import PCA


data = PCA(2).fit_transform(load().data)
res = gaussian_anomaly_detection(data)
colors = []
for x in res:
    if x < res.mean() - 2 * res.std() or x > res.mean() + 2 * res.std():
        colors.append('red')
    else:
        colors.append('green')
scatter(data[:,0], data[:,1], c=colors)
show()
hist(res, bins=100)
show()
