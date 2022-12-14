import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

def bias(y, y_pred):
    return np.mean((y - np.mean(y_pred)))**2

def variance(y_pred):
    return np.var(y_pred)

def bias_variance_tradeoff(model_type, model_name, max_degree, x_train, x_test, y_train, y_test):
    degrees = range(max_degree+1)
    num_bootstraps = 100
    biases = np.zeros(max_degree+1)
    variances = np.zeros(max_degree+1)
    mean_squared_errors = np.zeros(max_degree+1)
    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            model_type
        )
        boot_biases = np.zeros(num_bootstraps)
        boot_vars = np.zeros(num_bootstraps)
        boot_mses = np.zeros(num_bootstraps)
        for b in range(num_bootstraps):
            x, y = resample(x_train, y_train) # Bootstrap resample
            model.fit(x, y)
            y_pred = model.predict(x_test).flatten()
            boot_biases[b] = bias(y_test, y_pred)
            boot_vars[b] = variance(y_pred)
            boot_mses[b] = mean_squared_error(y_test, y_pred)

        biases[degree] = np.mean(boot_biases)
        variances[degree] = np.mean(boot_vars)
        mean_squared_errors[degree] = np.mean(boot_mses)

    plt.title(f'Bias-Variance Tradeoff for {model_name}')
    plt.xlabel('Degree')
    plt.plot(degrees, biases, label='Bias')
    plt.plot(degrees, variances, label='Variance')
    plt.plot(degrees, mean_squared_errors, label='Mean squared error')
    plt.legend()
    plt.savefig(f'./figures/bias_variance_tradeoff_{model_name}.pdf')
    plt.show()

if __name__ == '__main__':
    # Generate random samples
    N = 160
    np.random.seed(0)
    x = np.random.uniform(size=(N,)).reshape(-1,1) # For compatibility with Scikit-Learn
    y = lambda x: np.sin(1.0/x)
    
    # Split data into training set and test set
    train_size = 0.75
    i_split = int(train_size*x.size)
    x_train, x_test = x[:i_split], x[i_split:]
    np.random.seed(1)
    y_train = y(x_train) + np.random.normal(size=x_train.shape)
    y_test = y(x_test)
	
    # Generate plot of the bias-variance tradeoff for each model
    bias_variance_tradeoff(
        LinearRegression(fit_intercept=False), 
        'OLS',
        10,
        x_train, x_test, y_train, y_test
    )
    bias_variance_tradeoff(
        Ridge(fit_intercept=False), 
        'Ridge',
        30,
        x_train, x_test, y_train, y_test
    )
    bias_variance_tradeoff(
        Lasso(alpha=0.1, fit_intercept=False), 
        'Lasso',
        100,
        x_train, x_test, y_train, y_test
    )

