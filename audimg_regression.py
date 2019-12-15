# audimg_regression - continuous regression for auditory imagery stimulus encoding / decoding

import mvpa2.suite as P
import numpy as np
import sys

def _msg(*args, **kwargs):
    """
    Print arguments and flush stdout, optional argument newline [False]
    """
    newline = kwargs.setdefault('newline',False)
    for s in args:
        print s,
    if newline:
        print
    sys.stdout.flush()

def _standardize(x):
    """
    helper function 
    outputs:
        y=(x-m)/s - standardized vector, or column-wise standardization if x is a matrix
    """
    return (x - x.mean(0)) / x.std(0)

def _standardize_samples(ds):
    """
    z-score samples, with ds side effect
    returns:
        samples_mn, samples_std - z-score coefficients        
    """
    s = ds.samples
    samples_mn=s.mean(0)
    samples_std=s.std(0)
    ds.samples = _standardize(s)
    return samples_mn, samples_std


def _standardize_targets(ds):
    """
    z-score targets, with ds side effect
    returns:
        targets_mn, targets_std - z-score coefficients        
    """
    t = ds.targets
    targets_mn=t.mean(0)
    targets_std=t.std(0)
    ds.targets = _standardize(t)
    return targets_mn, targets_std

def _normalize_targets(ds):
    """
    center and normalize targets, with ds side effect
    returns:
        targets_mn, targets_max - normalization coefficients        
    """
    t = ds.targets
    targets_mn = t.mean(0)
    t = t - targets_mn
    targets_max = abs(t).max(0)
    ds.targets = t / targets_max
    return targets_mn, targets_max

def get_mapped_clf(clf, selector):
    svdmapper = P.SVDMapper()
    get_SVD_sliced = lambda x: P.ChainMapper([svdmapper, P.StaticFeatureSelection(x)])    
    selector = max(selector, 1)
    map_clf = P.MappedClassifier(clf, get_SVD_sliced(slice(0,selector)) )
    return map_clf

def _NU_SVR(ds, rbf, nu, gamma, n_select, postproc=None, errorfx=P.relative_rms_error, partitioner=P.HalfPartitioner(), **kwargs):
    """
    Helper function to compute _NU_SVR for dataset, given set of hyperparameters
    """
    if rbf:
        clf = P.SVM(svm_impl='NU_SVR', nu=nu, kernel=P.RbfSVMKernel(gamma=gamma), **kwargs)
    else:
        clf = P.SVM(svm_impl='NU_SVR', nu=nu, **kwargs)
    map_clf = get_mapped_clf(clf, selector=n_select)
    cv = P.CrossValidation(map_clf, partitioner, postproc=postproc, errorfx=errorfx, enable_ca=['stats'])
    res = cv(ds)
    return res

def optimal_NU_SVR(ds, rbf=True, calc_pred=False, **kwargs):
    """
    Get optimal NU_SVR for dataset
    """
    results = {}
    results['error'] = float('Inf')    
    for nu in np.linspace(.1,1,20):
        _msg(".")
        gammas = np.logspace(-1, -10, 20) if rbf else [1]
        for gamma in gammas:
            for n_select in np.logspace(0, np.log10(ds.shape[1]), 10):
                n_select = int(np.round(n_select))
                res = _NU_SVR(ds, rbf, nu, gamma, n_select)
                if res.samples.min() < results['error']:
                    results['error'] = res.samples.min()
                    results['nu'] = nu
                    results['gamma'] = gamma
                    results['nselect'] = n_select
                    results['res'] = res.samples
    _msg("nu=", results['nu'], "g=", results['gamma'], "n=", results['nselect'], "err=", round(results['error'],2),newline=1)
    results['prob']=_NU_SVR(ds, rbf, results['nu'], results['gamma'], results['nselect'], errorfx=P.corr_error_prob).samples
    if calc_pred:
        results['pred']=_NU_SVR(ds, rbf, results['nu'], results['gamma'], results['nselect'], errorfx=None)
    return results

def optimal_CLF(ds, clf=P.GPR(), calc_pred=False, partitioner=P.HalfPartitioner()):
    """
    get optimal CLF for dataset
    inputs:
         ds - dataset
        clf - classifier instance
    """
    results = {}
    results['error'] = float('Inf')    
    for n_select in np.logspace(0, np.log10(ds.shape[1]), 10):
        n_select = int(np.round(n_select))
        print n_select,
        sys.stdout.flush()
        map_clf = get_mapped_clf(clf, selector=n_select)
        cv = P.CrossValidation(map_clf, partitioner, postproc=None, errorfx=P.relative_rms_error, enable_ca=['stats'])
        res = cv(ds)
        if res.samples.min() < results['error']:
            results['error'] = res.samples.min()
            results['nselect'] = n_select
            results['res'] = res.samples
    _msg("n=", results['nselect'], "err=", round(results['error'],2),newline=1)
    map_clf = get_mapped_clf(clf, selector=results['nselect'])
    cv = P.CrossValidation(map_clf, partitioner, postproc=None, errorfx=P.corr_error_prob, enable_ca=['stats'])
    res = cv(ds)
    results['prob'] = res.samples
    if calc_pred:
        cv = P.CrossValidation(map_clf, partitioner, postproc=None, errorfx=None)
        res = cv(ds)
        results['pred'] = res.samples
    return results

def optimize_subj_model(ds, model='svr_rbf', condition=None, direction=None, **kwargs):
    """
    get optimal model for a subject
    inputs:
              ds  - masked dataset
            model - 'svr_lin', or 'svr_rbf'  ['svr_rbf']
              clf - clf instance for non-SVR models 
        condition - 0 (heard), 1 (imagined), or None (both) [None]
        direction - 0 (up), 1 (down), or None (both) [None]
         **kwargs - kwargs for get_subj_ds()
    outputs:
        results - dict of optimal model parameters & results
    """
    clf = kwargs.pop('clf', None)

    if clf is not None:
        results = optimal_CLF(ds, clf=clf)
    elif model=='svr_lin':
        results = optimal_NU_SVR(ds, rbf=False, tube_epsilon=0.01)
    elif model=='svr_rbf':
        results = optimal_NU_SVR(ds, rbf=True, tube_epsilon=0.01)
    else:
        raise ValueError("Unrecognized model type '{0}'".format(model))
    results['targets_z'] = ds.targets_z
    return results
