import numpy as np
import emcee

nside = 512

# Binning used for the power spectrum computation
lmin = 36.5
lmax = 300
ell = np.arange(lmin, lmax, 10)

# Discrad fraction for the MCMC fitting
discard_frac = 0.5

# Auto- and cross-spectra power law models for a three frequency bands fitting 
# theta are the parameter that we are fitting 
# freq are the central frequencies for each band used 
# cc are the colour corrections for each band 

def model_ee_bb_f1_f2_f3(theta, freq, cc, ell=ell):
    A_ee, A_bbee, alpha, betha, c_f1, c_f2, c_f3 = theta
    f1, f2, f3 = freq
    cc_f1, cc_f2, cc_f3 = cc

    # Definition of each Cl as a power law with a multipole pivot value of ell=80
    # and a frequency pivot value of 11.1 GHz
    cl_ee_f1 = A_ee * 1e-6 * (f1/11.1)**(2*betha) * (ell/80.)**alpha + c_f1 * 1e-9
    cl_bb_f1 = A_bbee * A_ee * 1e-6 * (f1/11.1)**(2*betha) * (ell/80.)**alpha + c_f1 * 1e-9

    cl_ee_f2 = A_ee * 1e-6 * (f2/11.1)**(2*betha) * (ell/80.)**alpha + c_f2 * 1e-9
    cl_bb_f2 = A_bbee * A_ee * 1e-6 * (f2/11.1)**(2*betha) * (ell/80.)**alpha + c_f2 * 1e-9

    cl_ee_f3 = A_ee * 1e-6 * (f3/11.1)**(2*betha) * (ell/80.)**alpha + c_f3 * 1e-9
    cl_bb_f3 = A_bbee * A_ee * 1e-6 * (f3/11.1)**(2*betha) * (ell/80.)**alpha + c_f3 * 1e-9

    return np.array([[cl_ee_f1/cc_f1**2, cl_bb_f1/cc_f1**2], [cl_ee_f2/cc_f2**2, cl_bb_f2/cc_f2**2], [cl_ee_f3/cc_f3**2, cl_bb_f3/cc_f3**2]])

def cross_model_ee_bb_f1_f2_f3(theta, freq, cc, ell=ell):
    A_ee, A_bbee, alpha, betha, c_f1, c_f2, c_f3 = theta
    f1, f2 = freq
    cc_f1, cc_f2 = cc

    # Definition of a cross-spectrum power law with the same pivot values as the auto-spectrum model
    cl_ee = A_ee * 1e-6 * (f1*f2/11.1**2)**(betha) * (ell/80.)**alpha 
    cl_bb = A_bbee * A_ee * 1e-6 * (f1*f2/11.1**2)**(betha) * (ell/80.)**alpha 

    return np.array([cl_ee / (cc_f1 * cc_f2), cl_bb / (cc_f1 * cc_f2)])


# =============================================================================
# Likeness function, priors setting, probability function and MCMC main 
# =============================================================================

# Likelihood function (accounts the chi-squared)
def lnlike_f1_f2_f3(theta, freq, cc, x, y_f1, y_f2, y_f3, y_f1_f2, y_f1_f3, y_f2_f3, yerr_f1, yerr_f2, yerr_f3, yerr_f1_f2, yerr_f1_f3, yerr_f2_f3):
    return -0.5 * (np.sum(((y_f1[0] - model_ee_bb_f1_f2_f3(theta, freq, cc)[0,0]) / yerr_f1[0])**2) + # EE f1xf1
                   np.sum(((y_f1[1] - model_ee_bb_f1_f2_f3(theta, freq, cc)[0,1]) / yerr_f1[1])**2) + # BB f1xf1
                   np.sum(((y_f2[0] - model_ee_bb_f1_f2_f3(theta, freq, cc)[1,0]) / yerr_f2[0])**2) + # EE f2xf2
                   np.sum(((y_f2[1] - model_ee_bb_f1_f2_f3(theta, freq, cc)[1,1]) / yerr_f2[1])**2) + # BB f2xf2
                   np.sum(((y_f3[0] - model_ee_bb_f1_f2_f3(theta, freq, cc)[2,0]) / yerr_f3[0])**2) + # EE f3xf3
                   np.sum(((y_f3[1] - model_ee_bb_f1_f2_f3(theta, freq, cc)[2,1]) / yerr_f3[1])**2) + # BB f3xf3
                   np.sum(((y_f1_f2[0] - cross_model_ee_bb_f1_f2_f3(theta, [freq[0],freq[1]], [cc[0],cc[1]])[0]) / yerr_f1_f2[0])**2) + # EE f1xf2
                   np.sum(((y_f1_f2[1] - cross_model_ee_bb_f1_f2_f3(theta, [freq[0],freq[1]], [cc[0],cc[1]])[1]) / yerr_f1_f2[1])**2) + # BB f1xf2
                   np.sum(((y_f1_f3[0] - cross_model_ee_bb_f1_f2_f3(theta, [freq[0],freq[2]], [cc[0],cc[2]])[0]) / yerr_f1_f3[0])**2) + # EE f1xf3
                   np.sum(((y_f1_f3[1] - cross_model_ee_bb_f1_f2_f3(theta, [freq[0],freq[2]], [cc[0],cc[2]])[1]) / yerr_f1_f3[1])**2) + # BB f1xf3
                   np.sum(((y_f2_f3[0] - cross_model_ee_bb_f1_f2_f3(theta, [freq[1],freq[2]], [cc[1],cc[2]])[0]) / yerr_f2_f3[0])**2) + # EE f2xf3
                   np.sum(((y_f2_f3[1] - cross_model_ee_bb_f1_f2_f3(theta, [freq[1],freq[2]], [cc[1],cc[2]])[1]) / yerr_f2_f3[1])**2))  # BB f2xf3


# Priors setting 
def lnprior_f1_f2_f3(theta):
    A_ee, A_bbee, alpha, betha, c_f1, c_f2, c_f3 = theta
    if -6 < alpha < -0.5 and -5 < betha < -1: # and c_f1 > 0 and c_f2 > 0 and c_f3 > 0:
        return 0.0
    return -np.inf

# Log probability function (accounting priors)
def lnprob_f1_f2_f3(theta, freq, cc, x, y_f1, y_f2, y_f3, y_f1_f2, y_f1_f3, y_f2_f3, yerr_f1, yerr_f2, yerr_f3, yerr_f1_f2, yerr_f1_f3, yerr_f2_f3):
    lp = lnprior_f1_f2_f3(theta)
    if not np.isfinite(lp): # If the parameter is not within the priors, return a -infinite
        return -np.inf
    return lp + lnlike_f1_f2_f3(theta, freq, cc, x, y_f1, y_f2, y_f3, y_f1_f2, y_f1_f3, y_f2_f3, yerr_f1, yerr_f2, yerr_f3, yerr_f1_f2, yerr_f1_f3, yerr_f2_f3)

# Main MCMC sampler

def main(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    # Burn-in process, discarding the first discard_frac of the chain
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, int(niter*discard_frac), progress=True) 
    sampler.reset()

    # Final posteriors 
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True) # Run niter iterations with the new initial positions

    return sampler, pos, prob, state  


# =============================================================================
# Walkers, iteartions and initial parameters
# =============================================================================

nwalkers = 100 # Number of chains
niter = 10000 # Number of Monte Cralo realizations
ndim = 7 # Number of parameters
initial= [1.5, 0.3, -3., -3., 0., 0., 0.] # Initial values for each parameters
variations = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]) # Random variations for the initial values
p0 = [np.array(initial) + variations * np.random.randn(ndim) for i in range(nwalkers)]

# data must be an list or array including: 
# [ [freq_1, freq_2, freq_3], [cc_f1, cc_f2, cc_f3], ell, 
# [cl_f1_ee, cl_f1_bb], [cl_f2_ee, cl_f2_bb], [cl_f3_ee, cl_f3_bb], 
# [cross_f1_f2_ee, cross_f1_f2_bb], [cross_f1_f3_ee, cross_f1_f3_bb], [cross_f2_f3_ee, cross_f2_f3_bb], 
# [error_cl_f1_ee, error_cl_f1_bb], [error_cl_f2_ee, error_cl_f2_bb], [error_cl_f3_ee, error_cl_f3_bb], 
# [error_cross_f1_f2_ee, error_cros_f1_f2_bb], [error_cross_f1_f3_ee, error_cros_f1_f3_bb], [error_cross_f2_f3_ee, error_cros_f2_f3_bb] ]
data = [] 

# Final posteriors for each parameter, can be plotted as a corner figure
sampler, _, _, _ = main(p0, nwalkers, niter, ndim, lnprob_f1_f2_f3, data)
    
