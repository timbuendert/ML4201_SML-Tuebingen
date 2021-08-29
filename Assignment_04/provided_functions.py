import numpy as np
import matplotlib.pyplot as plt


def LassoObjective(wplus, wminus, Phi, Y, lmbd):
    ''' evaluates the objective function at (wplus, wminus)
    L2 loss + L1 regularization
    '''
    w = wplus - wminus
    return ((Phi @ w - Y) ** 2).mean(
        ) + lmbd * np.abs(w).sum()


def GradLassoObjective(wplus, wminus, Phi, Y, lmbd):
    ''' computes the gradients of the objective function
    at (wplus, wminus)
    gradwplus: gradient wrt wplus
    gradwminus: gradient wrt minus

    FILL IN
    '''
    return gradwplus, gradwminus


def ProjectionPositiveOrthant(x):
    ''' returns the projection of x onto the positive orthant

    FILL IN
    '''
    return y


def getStepSize(wplus, wminus, Phi, Y, lmbd, gradwplus,
                gradwminus, loss):
    ''' performs one step of projected gradient descent (i.e.
    compute next iterate) with step size selection via
    backtracking line search

    input
    loss: objective function at current iterate (wplus, wminus)

    output
    wplusnew, wminusnew: next iterates wplus_{t+1}, wminus_{t+1}
    lossnew: objective function at the new iterate
    
    FILL IN
    '''
    alpha, beta, sigma = 1., .1, .1
    wplusnew, wminusnew = wplus.copy(), wminus.copy()
    lossnew = np.float('Inf') # make sure to enter the loop

    # choose the step size alpha with backtracking line search
    while lossnew > loss + sigma * ((gradwplus * (
        wplusnew - wplus)).sum() + (gradwminus * (
        wminusnew - wminus)).sum()):
        # get new step size to test
        alpha *= beta

        # projected gradient step for wplus and wminus with step size alpha
        # i.e. compute x_{t+1} as in the text
        # FILL IN
        print('fill in with projected gradient step')

        # compute new value of the objective
        lossnew = LassoObjective(wplusnew, wminusnew, Phi, Y, lmbd)

    return wplusnew, wminusnew, lossnew


def Lasso(Phi, Y, lmbd):
    ''' compute weight of linear regression with Lasso

    Phi: deisgn matrix n x d
    Y: true values n x 1
    lmbd: weight of regularization

    output: weights of linear regression d x 1
    '''
    # initialize wplus, wminus
    wplus = np.random.rand(Phi.shape[1], 1)
    wminus = np.random.rand(*wplus.shape)
    loss = LassoObjective(wplus, wminus, Phi, Y, lmbd)

    counter = 1
    while counter > 0:
        # compute gradients wrt wplus and wminus
        gradwplus, gradwminus = GradLassoObjective(
            wplus, wminus, Phi, Y, lmbd)

        # compute new iterates
        wplus, wminus, loss = getStepSize(wplus,
            wminus, Phi, Y, lmbd, gradwplus, gradwminus, loss)

        if (counter % 100) == 0:
            # check if stopping criterion is met
            wnew = wplus - wminus
            ind = wnew != 0.
            indz = wnew == 0.
            r = 2 / Phi.shape[0] * (Phi.T @ (Phi @ wnew - Y))
            stop = np.abs(r[ind] + lmbd * np.sign(wnew[ind]
                )).sum() + (np.abs(r[indz]) - lmbd * np.ones_like(
                r[indz])).clip(0.).sum()
            print('iter={} current objective={:.3f} nonzero weights={}'.format(
                counter, loss, ind.sum()) +\
                ' stop={:.5f}'.format(stop / Phi.shape[0]))
            if np.abs(stop) / Phi.shape[0] < 1e-5:
                break
        counter += 1

    return wplus - wminus

