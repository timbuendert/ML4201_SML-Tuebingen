import numpy as np


def CoordinateDescentSVM(Xtrain, Ytrain, C, Xtest, Ytest):
    ''' compute the solution of linear SVM
    Xtrain, Ytrain: training set
    Xtest, Ytest: test set (only to monitor test error)
    C: error parameter

    w: weights linear SVM
    TrainError, TestError: training and test errors over iterations
    '''
    n = Xtrain.shape[0]
    alpha = np.zeros([n, 1]) # initialize dual variables
    w = Xtrain.T @ (Ytrain * alpha) # initialize primal variables

    counter = 0
    converged = False
    eps = 1e-3
    TrainError, TestError = [], []
    
    while not converged:
        # select coordinate to update
        r = counter % n

        # solve the subproblem for coordinate r without any constraints
        # FILL IN

        # project the solution to the interval [0, C / n]
        # FILL IN

        # monitor the progress of the method computing the dual
        # objective DualObj
        # FILL IN

        if (counter + 1) % 100 == 0:
            print('iteration={} dual obj={:.3f}'.format(
                counter + 1, DualObj))

        # compute the training and test error with the current iterate alpha
        # FILL IN

        # if the KKT conditions are satisfied up to the tolerance eps by the
        # the current iterate alpha then set converged = True
        # FILL IN

        counter += 1
        
    # compute the primal solution w from alpha
    # FILL IN

    # show final dual objective
    print('final iteration={} dual obj={:.3f}'.format(counter, DualObj))

    return w, TrainError, TestError

