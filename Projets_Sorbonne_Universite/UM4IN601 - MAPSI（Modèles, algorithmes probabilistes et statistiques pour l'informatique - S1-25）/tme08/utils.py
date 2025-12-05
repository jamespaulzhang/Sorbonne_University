import numpy as np
def gen_data_lin(a, b, sig, N, Ntest):
    X_train = np.sort(np.random.rand(N))
    X_test  = np.sort(np.random.rand(Ntest))
    y_train = a*X_train+b+np.random.randn(N)*sig
    y_test  = a*X_test+b+np.random.randn(Ntest)*sig
    return X_train, y_train, X_test, y_test