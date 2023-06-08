"""
Implements "HMM-EM" method from:

Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown;
Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.
doi: https://doi.org/10.1162/neco.2009.06-08-799

Translated to python 05/22 by Tom Bugnon from MATLAB code provided by Zhe Sage Chen

Differences from original MATLAB code: 
- Use all bins with shorter window for history at beginning
- Use numpy RNG in newton_ralphson (So different output as MATLAB) ( TODO: https://stackoverflow.com/a/36823993 )
- Max number of iterations is params['n_iter_EM'] rather than n_iter_EM - 1
- normalize "bin_history_spike_count"
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import factorial

from .. import utils
from .exceptions import NumericalErrorException, FailedInitializationException

HMMEM_PARAMS = {
    "binsize": 0.010,  # (s) (Discrete algorithm)
    "history_window_nbins": 3,  # Size of history window IN BINS
    "n_iter_EM": 200,  # Number of iterations for EM
    "n_iter_newton_ralphson": 100,
    "init_A": np.array(
        [[0.1, 0.9], [0.01, 0.99]]
    ),  # Initial transition probability matrix
    "init_state_estimate_method": "liberal",  # Method to find inital OFF states to fit GLM model with. Ignored if init_mu/alphaa/betaa are specified. Either of 'conservative'/'liberal'/'intermediate'
    "init_mu": None,  # ~ OFF rate. Fitted to data if None
    "init_alphaa": None,  # ~ difference between ON and OFF rate. Fitted to data if None
    "init_betaa": None,  # ~ Weight of recent history firing rate. Fitted to data if None,
    "gap_threshold": None,  # Merge active states separated by less than gap_threhsold
}


# TODO: Nx1-array for betaa (one value per "history window")
def run_hmmem(
    train,
    Tmax,
    params,
    output_dir=None,
    filename=None,  # TODO harmonize
    save=None,  # TODO harmonize
    verbose=True,
):
    # Params
    assert set(params.keys()) == set(HMMEM_PARAMS.keys())

    # Merge and bin all trains
    bins = np.arange(0, Tmax + params["binsize"], params["binsize"])
    nbins = len(bins)
    bin_spike_count, _ = np.histogram(train, bins)
    bin_spike_count = bin_spike_count.astype(int)

    # History spike count of each bin (through panda roll)
    # Substract bin_spike_count to exclude current bin
    # (while keeping actual window of size X), so X+1 -1
    # (Stay consistent with matlab)
    # TODO: Modify for N-dim betaa
    bin_history_spike_count = (
        pd.Series(bin_spike_count)
        .rolling(
            params["history_window_nbins"] + 1,  # Sum over window of N bins
            center=False,  # Window left of each sample
            min_periods=1,  # Sum over fewer bins at beginning of array (Unused if we trim)
        )
        .sum()
        .to_numpy(dtype=int)
        - bin_spike_count
    )
    # Normalize (avoid overflows with large history window)
    bin_history_spike_count = bin_history_spike_count / params["history_window_nbins"]

    # Reshape to 1xnbins vectors (MATLAB consistency of _run_hmmem)
    bin_spike_count = bin_spike_count.reshape((1, -1))
    bin_history_spike_count = bin_history_spike_count.reshape((1, -1))

    # Ignore bins without full history
    bin_spike_count_trimmed = bin_spike_count[:, params["history_window_nbins"] :]
    bin_history_spike_count_trimmed = bin_history_spike_count[
        :, params["history_window_nbins"] :
    ]

    # Fit init_alphaa, init_mu and init_betaa ?
    if all([params[p] is None for p in ["init_alphaa", "init_mu", "init_betaa"]]):
        fitted_init_params = True
        init_mu, init_alphaa, init_betaa = fit_init_poisson_params(
            bin_spike_count_trimmed[0, :],
            bin_history_spike_count_trimmed[0, :],
            params["binsize"],
            init_state_estimate_method=params.get("init_state_estimate_method", None),
        )
    else:
        if any([params[k] is None for k in ["init_alphaa", "init_mu", "init_betaa"]]):
            raise ValueError(
                "'init_alphaa', 'init_mu' and 'init_betaa' params should either be all floats or all None."
            )
        fitted_init_params = False
        init_alphaa = params["init_alphaa"]
        init_mu = params["init_mu"]
        init_betaa = params["init_betaa"]

    (
        S,
        prob_S,
        alphaa,
        betaa,
        mu,
        A,
        B,
        p0,
        log_L,
        log_P,
        end_iter_EM,
        EM_converged,
    ) = _run_hmmem(
        bin_spike_count_trimmed,
        bin_history_spike_count_trimmed,
        np.array(params["init_A"]),
        init_alphaa,
        init_betaa,
        init_mu,
        int(params["n_iter_EM"]),
        int(params["n_iter_newton_ralphson"]),
        verbose=verbose,
    )

    # Return identical result from MATLAB original code
    # return S, prob_S, alphaa, betaa, mu, A, B, p0, log_L, log_P

    # 1d-Array of active/inactive bins
    # Broadcast trimmed data to original number of bins
    active_bin = S[0, 0] * np.ones(
        (nbins,)
    )  # Set same value for ignored bins at the beginning as first detected value
    active_bin[params["history_window_nbins"] + 1 :] = S[0, :]  # 1d
    assert all([s in [0, 1] for s in active_bin])
    srate = 1 / params["binsize"]  # "Sampling rate" of returned binned states (Hz)

    ## Remove short OFF states and return as pd.Dataframe

    # Merge active states separated by less than gap_threshold
    if params["gap_threshold"] is not None and params["gap_threshold"] > 0:
        if verbose:
            print("Merge closeby on-periods...", end="")
        off_durations = utils.state_durations(active_bin, 0, srate=srate)
        off_starts = utils.state_starts(active_bin, 0)
        off_ends = utils.state_ends(active_bin, 0)
        N_merged = 0
        for i, off_dur in enumerate(off_durations):
            if off_dur <= params["gap_threshold"]:
                active_bin[off_starts[i] : off_ends[i] + 1] = 1
                N_merged += 1
        if verbose:
            print(f"Merged N={N_merged} active periods")

    # Return df
    # all in (sec)
    if verbose:
        print("Get final on/off periods df...")
    on_starts = utils.state_starts(active_bin, 1) / srate
    off_starts = utils.state_starts(active_bin, 0) / srate
    on_ends = utils.state_ends(active_bin, 1) / srate
    off_ends = utils.state_ends(active_bin, 0) / srate
    on_durations = utils.state_durations(
        active_bin,
        1,
        srate=srate,
    )
    off_durations = utils.state_durations(
        active_bin,
        0,
        srate=srate,
    )
    N_on = len(on_starts)
    N_off = len(off_starts)

    # TODO: Return _run_hmmem info bin by bin?
    N_on_off = N_on + N_off
    on_off_df = pd.DataFrame(
        {
            "state": ["on" for i in range(N_on)] + ["off" for i in range(N_off)],
            "start_time": list(on_starts) + list(off_starts),
            "end_time": list(on_ends) + list(off_ends),
            "duration": list(on_durations) + list(off_durations),
            "cumFR": len(train) / Tmax,
            "alphaa": alphaa,
            "betaa": [betaa] * N_on_off,
            "mu": mu,
            "A": [A] * N_on_off,
            "log_L": log_L,
            "end_iter_EM": end_iter_EM,
            "EM_converged": EM_converged,
            # **{
            #     k: [v] * N_on_off for k, v in params.items()
            # },
        }
    )
    # if fitted_init_params:
    #     # Save fitted params actually used to initialize HMMEM
    #     on_off_df['init_mu_fitted'] = init_mu
    #     on_off_df['init_alphaa_fitted'] = init_alphaa
    #     on_off_df['init_betaa_fitted'] = init_betaa
    # else:
    #     on_off_df['init_mu_fitted'] = None
    #     on_off_df['init_alphaa_fitted'] = None
    #     on_off_df['init_betaa_fitted'] = None

    return on_off_df.sort_values(by="start_time").reset_index(drop=True)


# TODO: Nx1-array for betaa (one value per "history window")
def _run_hmmem(
    bin_spike_count,
    bin_history_spike_count,
    init_A,
    init_alphaa,
    init_betaa,
    init_mu,
    n_iter_EM,
    n_iter_newton_ralphson,
    verbose=True,
):

    # Param check
    bin_spike_count = np.atleast_1d(bin_spike_count).astype(int)
    bin_history_spike_count = np.atleast_1d(bin_history_spike_count).astype(int)
    assert bin_spike_count.shape[0] == 1  # Horizontal vectors
    assert bin_spike_count.shape[1] == bin_history_spike_count.shape[1]
    # TODO: bin_history_spike_count could be dimension (k, nbins) rather than (1, nbins)
    #   if so beta would be dimension k
    if not isinstance(init_betaa, (float, int)):
        raise NotImplementedError()
    if bin_history_spike_count.shape[0] > 1:
        raise NotImplementedError()

    # Constants
    EPS = np.spacing(1)
    STATES = [0, 1]

    # Output vars
    A = init_A.copy()
    alphaa = init_alphaa
    betaa = init_betaa
    mu = init_mu

    nbins = bin_spike_count.shape[1]

    # Eq 2.3
    bin_lambda = np.zeros((2, nbins), dtype=float)
    for k in range(nbins):
        for state_i in range(len(STATES)):
            bin_lambda[state_i, k] = np.exp(
                mu
                + alphaa * STATES[state_i]
                + betaa * bin_history_spike_count[0, k]  # TODO if betaa not scalar
            )

    # Eq 2.2
    B = np.empty((2, nbins), dtype=float)
    for state_i in range(len(STATES)):
        B[state_i, :] = (
            np.exp(-bin_lambda[state_i, :])
            * np.power(bin_lambda[state_i, :], bin_spike_count)
            / factorial(bin_spike_count)
        )

    p0 = 0.5 * np.ones((2,))
    # alpha[k]: forward message of state i at time k
    # beta[k]: backward message of state i at time k
    # gamma[k]: marginal conditional probability at time k: P(Sk = i | H)
    # zeta[i, j, k]:  joint conditional probability: P(Sk-1 = i , Sk = j | H)
    alpha, beta, gamma = [np.zeros((2, nbins), dtype=float) for _ in range(3)]
    zeta = np.zeros((2, 2, nbins), dtype=float)

    # Forward-backward E-M algorithm
    t = 0
    log_P = np.empty((n_iter_EM,), dtype=float)
    diff_log_P = 10
    while (
        t < n_iter_EM and diff_log_P > 0
    ):  # Differ from original MATLAB algo here (it uses n_iter_EM - 1)

        ## E-step: forward algorithm

        # Compute alpha (k=0)
        C = np.zeros((nbins,))  # Scaling vector to avoid numerical inaccuracies
        alpha[:, 0] = p0 * B[:, 0]
        C[0] = np.sum(alpha[:, 0], axis=None)
        alpha[:, 0] = alpha[:, 0] / C[0]  # Scaling
        for k in range(1, nbins):
            # Compute alpha (k > 0)
            alpha[:, k] = np.multiply(
                np.matmul(alpha[:, k - 1].transpose(), A), B[:, k]
            )
            C[k] = np.sum(alpha[:, k], axis=None)
            if C[k] > 0:
                alpha[:, k] = alpha[:, k] / C[k]
            else:
                raise NumericalErrorException(
                    f"Numerical error when scaling alpha[k] "
                    f"(EM step t = {t}, bin index k = {k})"
                )

        log_P[t] = np.sum(np.log(C[0:nbins] + EPS))

        ## E-step: backward algorithm

        # Compute beta
        beta[:, -1] = 1
        beta[:, -1] = beta[:, -1] / C[-1]
        for k in range(nbins - 2, -1, -1):
            beta[:, k : k + 1] = np.matmul(
                A, np.multiply(beta[:, k + 1 : k + 2], B[:, k + 1 : k + 2])
            )
            beta[:, k] = beta[:, k] / C[k]

        for k in range(0, nbins - 1):
            temp = np.multiply(
                np.matmul(
                    alpha[:, k : k + 1],
                    np.multiply(
                        beta[:, k + 1 : k + 2], B[:, k + 1 : k + 2]
                    ).transpose(),
                ),
                A,
            )
            if np.sum(temp, axis=None) > 0:
                zeta[:, :, k] = temp / np.sum(temp, axis=None)
            else:
                raise NumericalErrorException(
                    f"Numerical error when scaling zeta_i_j[k] "
                    f"(EM step t = {t}, bin index k = {k})"
                )
            gamma[:, k] = np.sum(zeta[:, :, k], axis=1)

        # Sufficient statistics
        # Z = E[S]
        Z = STATES[0] * gamma[0, :] + STATES[1] * gamma[1, :]

        ## M-step:

        # Update transition matrix
        p0 = gamma[:, 0]
        temp1 = np.sum(zeta, axis=2)
        temp2 = np.sum(gamma, axis=1, keepdims=True)
        A = temp1 / np.tile(temp2, (1, 2))

        # Update alphaa/betaa/mu
        alphaa, betaa, mu = newton_ralphson(
            bin_spike_count,
            Z,
            bin_history_spike_count,
            alphaa,
            betaa,
            mu,
            n_iter_newton_ralphson,
        )

        # Update lambda cif
        for k in range(nbins):
            for state_i in range(len(STATES)):
                bin_lambda[state_i, k] = np.exp(
                    mu
                    + alphaa * STATES[state_i]
                    + betaa * bin_history_spike_count[0, k]  # TODO if betaa not scalar
                )

        # Update B
        for state_i in range(len(STATES)):
            B[state_i, :] = (
                np.exp(-bin_lambda[state_i, :])
                * np.power(bin_lambda[state_i, :], bin_spike_count)
                / factorial(bin_spike_count)
            )

        # Verbose and increment
        if verbose:
            print(
                f"n_iter_EM={t}, log-likelihood={log_P[t]}, mu={mu}, alpha={alphaa}, beta={betaa}, A={A}"
            )
        if t > 1:
            diff_log_P = log_P[t] - log_P[t - 1]
        t += 1

        ## End E-M Loop

    end_iter_EM = t - 1  # (Since we just incremented)
    EM_converged = t < n_iter_EM  # (idem)

    prob_S = gamma  # 2 x nbins
    mean_S = (
        STATES[0] * gamma[0:1, :] + STATES[1] * gamma[1:, :]
    )  # Expected mean, 1 x nbins
    p0 = gamma[:, 0:1]  # 2 x 1

    # Viterbi algorithm for decoding most likely states
    delta = np.zeros((2, nbins))
    psi = np.zeros((2, nbins), dtype=int)
    S = np.zeros((1, nbins), dtype=int)

    # Working in log-domain
    log_P0 = np.log(p0 + EPS)
    logA = np.log(A + EPS)
    logB = np.log(B + EPS)

    delta[:, 0:1] = log_P0 + logB[:, 0:1]  # 2 x 1
    psi[:, 1:2] = 0

    for t in range(1, nbins):
        # Maximum and argmax of every row (col?)
        temp = np.transpose(np.matmul(delta[:, t - 1 : t], np.ones((1, 2))) + logA)
        # psi[:, t:t+1] = np.argmax(temp, axis=1, keepdims=True)
        psi[:, t : t + 1] = np.expand_dims(
            np.argmax(temp, axis=1), axis=1
        )  # no keepdims kwarg for numpy <= 1.21
        # delta[:, t:t+1] = np.max(temp, axis=1, keepdims=True) + logB[:, t:t+1]
        delta[:, t : t + 1] = (
            np.expand_dims(np.max(temp, axis=1), axis=1) + logB[:, t : t + 1]
        )  # no keepdims kwarg for numpy <= 1.21

    # State estimate
    S[0, -1] = np.argmax(delta[:, -1], axis=None)
    log_L = (
        delta[S[0, -1], -1] / nbins
    )  # TODO: Is this correct? Where is likelihood of each state?
    for t in range(nbins - 2, -1, -1):
        S[0, t] = psi[S[0, t + 1], t + 1]

    p0 = np.exp(p0)
    if not set(STATES) == set([0, 1]):
        raise NotImplementedError

    return (
        S,
        prob_S,
        alphaa,
        betaa,
        mu,
        A,
        B,
        p0,
        log_L,
        log_P,
        end_iter_EM,
        EM_converged,
    )


def get_initial_state_estimate(
    bin_spike_count,
    binsize,
    method="liberal",
    off_min_duration=0.05,
    on_min_duration=0.01,
):
    """Compute initial estimate of ON/OFF states.

    Used to fit GLM parameters and initialize HMMEM algorithm.
    Ideally we'd use a ground truth.

    Here we initialize assuming OFF bins are those with:
        - minimal count for more than 50msec (`method`=='conservative')
        - almost minimal count for more than 50msec (`method`=='liberal')
        - minimal count for more than 50msec, allowing for 10msec of
            almost minimal count
    """
    if method is None:
        method = "liberal"
    assert method in ["liberal", "intermediate", "conservative"]

    if binsize > 0.010:
        raise NotImplementedError()

    def _flip_short_periods(active_bin, state, min_duration, binsize):
        assert state in [0, 1]
        srate = 1 / binsize
        state_durations = utils.state_durations(active_bin, state, srate=srate)  # (sec)
        state_starts = utils.state_starts(active_bin, state)
        state_ends = utils.state_ends(active_bin, state)
        for i, dur in enumerate(state_durations):
            if dur <= min_duration:
                active_bin[state_starts[i] : state_ends[i] + 1] = not state
        return active_bin

    if method == "liberal":
        active_bin = (bin_spike_count > min(bin_spike_count) + 1).astype(bool)
    elif method == "conservative":
        active_bin = (bin_spike_count > min(bin_spike_count)).astype(bool)
    elif method == "intermediate":
        active_bin = (bin_spike_count > min(bin_spike_count)).astype(bool)
        # Remove very short ONs
        active_bin = _flip_short_periods(active_bin, 1, on_min_duration, binsize)

    # Remove short off periods
    active_bin = _flip_short_periods(active_bin, 0, off_min_duration, binsize)

    if all(active_bin):
        # Found only ONs
        raise FailedInitializationException()
    elif not any(active_bin):
        # Found only OFFs
        raise FailedInitializationException()

    return active_bin.astype(int)


def fit_init_poisson_params(
    bin_spike_count,
    bin_history_spike_count,
    binsize,
    init_state_estimate_method="liberal",
):  # 1D arrays

    ## Result
    endog = pd.DataFrame({"count": bin_spike_count})

    ## Predictors
    state = get_initial_state_estimate(
        bin_spike_count,
        binsize,
        method=init_state_estimate_method,
    )
    exog = sm.add_constant(
        pd.DataFrame(
            {
                # "state": np.zeros((nbins,)),
                "state": state,
                "history": bin_history_spike_count,
            }
        )
    )
    # count ~ mu + alphaa * state + betaa * bin_history
    mod = sm.GLM(endog, exog, family=sm.families.Poisson(link=sm.families.links.log()))
    res = mod.fit()

    print(res.summary())

    return res.params.values


def newton_ralphson(
    bin_spike_count,
    Z,
    bin_history_spike_count,
    init_alphaa,
    init_betaa,
    init_mu,
    n_iter,
):

    assert bin_spike_count.shape[1] > bin_spike_count.shape[0]  # Horizontal
    assert (
        bin_history_spike_count.shape[1] > bin_history_spike_count.shape[0]
    )  # Horizontal

    if bin_history_spike_count.shape[0] > 1:
        raise NotImplementedError()

    mu, alphaa = 0.0, 0.0
    if not isinstance(init_betaa, float):
        raise NotImplementedError()
    else:
        betaa = 0.0

    # Update mu
    temp0 = np.sum(bin_spike_count, axis=None)
    update = init_mu + 0.01 * np.random.randn()  # Differ from MATLAB output here
    for _ in range(n_iter):
        g = (
            np.sum(
                np.exp(
                    update
                    + init_alphaa * Z
                    + init_betaa * bin_history_spike_count
                    # # If betaa is not scalar (use np.dot?)
                    # update + init_alphaa * Z \
                    #     + np.matmul(
                    #         init_betaa.transpose(),
                    #         bin_history_spike_count
                    #     )
                )
            )
            - temp0
        )
        gprime = np.sum(
            np.exp(
                update
                + init_alphaa * Z
                + init_betaa * bin_history_spike_count
                # # # If betaa is not scalar (use np.dot?)
                # update + init_alphaa * Z \
                #     + np.matmul(
                #         init_betaa.transpose(),
                #         bin_history_spike_count
                #     )
            )
        )  # Derivative w.r.t update
        update = update - g / gprime
        mu = update

    # Update alphaa
    temp1 = np.sum(bin_spike_count * Z, axis=None)
    update = init_alphaa + 0.01 * np.random.randn()  # Differ from MATLAB output here
    for _ in range(n_iter):
        g = (
            np.sum(
                Z
                * np.exp(
                    mu
                    + update * Z
                    + init_betaa * bin_history_spike_count
                    # # If betaa is not scalar (use np.dot?)
                    # mu + update * Z \
                    #     + np.matmul(
                    #         init_betaa.transpose(),
                    #         bin_history_spike_count
                    #     )
                )
            )
            - temp1
        )
        gprime = np.sum(
            Z
            * Z
            * np.exp(
                mu
                + update * Z
                + init_betaa * bin_history_spike_count
                # # If betaa is not scalar (use np.dot?)
                # mu + update * Z \
                #     + np.matmul(
                #         init_betaa.transpose(),
                #         bin_history_spike_count
                #     )
            )
        )  # Derivative w.r.t update
        update = update - g / gprime

        alphaa = update

    # Update betaa
    d = bin_history_spike_count.shape[0]
    if d == 1:
        temp2 = np.sum(bin_spike_count * bin_history_spike_count, axis=None)
        update = init_betaa + 0.01 * np.random.randn()  # Differ from MATLAB output here
        for _ in range(n_iter):
            g = (
                np.sum(
                    bin_history_spike_count
                    * np.exp(mu + alphaa * Z + update * bin_history_spike_count)
                )
                - temp2
            )
            gprime = np.sum(
                bin_history_spike_count
                * bin_history_spike_count
                * np.exp(mu + alphaa * Z + update * bin_history_spike_count)
            )  # Derivative w.r.t update
            update = update - g / gprime

            betaa = update
    elif d >= 1:
        # Corresponding piece of matlab code from Zhe Sage Chen:
        # %[d,ydim] = size(count);
        # %[1,ydim] = size(Y);
        # tem2 = sum(repmat(Y,d,1) .* count, 2);
        # update = beta_old + 0.01*randn(d,1);
        # for i = 1:Iter
        #     g = sum(count .* repmat( exp(mu_new + alpha_new * Z + update' * count), d, 1), 2) - tem2;  % vector
        #     gprime = count .* repmat( exp(mu_new + alpha_new * Z + update' * count), d, 1) * count';   % matrix
        #     update = update - inv(gprime) * g;

        #     beta_new = update;
        # end
        raise NotImplementedError()

    return alphaa, betaa, mu
