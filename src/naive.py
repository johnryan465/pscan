def naive_pscan(A, X, Y_init):
    y = Y_init
    s = 0

    for k in range(A.size(1)):
        y = A[:, k, None] * y + X[:, k]
        s = s + y
    Y_ = s
    return Y_

fn = naive_pscan