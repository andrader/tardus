import numbers
from typing import Any

import sklearn.utils._param_validation as pv
from sklearn.base import BaseEstimator


def normalize_constraints(constraints):
    c_expandable = (
        pv.MissingValues,
        pv._RandomStates,
        pv._CVObjects,
        pv._VerboseHelper,
    )

    c_supported = (pv._Booleans, pv.Interval, pv.Options, pv._IterablesNotString)
    c_types_supported = (str, bool, numbers.Integral, numbers.Real, dict)

    constraints = [pv.make_constraint(c) for c in constraints]
    constraints = [
        c._constraints if isinstance(c, c_expandable) else [c] for c in constraints
    ]
    constraints = [c for constraint_list in constraints for c in constraint_list]

    ignored = [
        c
        for c in constraints
        if not (isinstance(c, c_supported) or _isinstance_of(c, c_types_supported))
    ]
    constraints = [
        c
        for c in constraints
        if (isinstance(c, c_supported) or _isinstance_of(c, c_types_supported))
    ]

    return constraints, ignored


def _isinstance_of(c, type):
    return isinstance(c, pv._InstancesOf) and issubclass(c.type, type)


def get_params_constraints(estimator: BaseEstimator) -> dict[str, tuple[list, Any]]:
    # initiate if not
    estimator = estimator if isinstance(estimator, BaseEstimator) else estimator()

    params_defaults = estimator.get_params()
    constraints = estimator._parameter_constraints

    norm_constraints = {}
    ignored_constraints = {}
    for k, clist in constraints.items():
        norm_clist, ignored = normalize_constraints(clist)
        norm_constraints[k] = norm_clist
        ignored_constraints[k] = ignored

    # sigparams = inspect.signature(estimator.__init__).parameters
    # defaults = {k: sigparams[k].default for k in params}
    # # overwrite signature defaults with instance params
    # params = dict(defaults, **params)

    # estimator_kwargs = estimator.get_params()
    # params = func_sig.bind(**estimator_kwargs)
    # params.apply_defaults()

    results = {
        k: (
            norm_constraints.get(k, list()),
            params_defaults[k],
            ignored_constraints.get(k),
        )
        for k in params_defaults
    }

    return results
