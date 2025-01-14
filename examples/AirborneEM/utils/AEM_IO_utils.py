import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from .AEM_plot_utils import *
import arviz as az
import corner

# read in enembles
def read_aem_ensembles(dir):
    log_posterior_ens,ensemble_per_state,ndims = [],[],[]
    for i in range(10):
        chain=read_chain(dir+'/lyr{:02d}/chain.out'.format(i+1))
        n,msft,d,logc=chain[0]
        ndims.append(len(np.concatenate((d,logc)))-1)
        y=[]
        x = np.zeros((len(chain),ndims[i]))
        for j,piece in enumerate(chain):
            n,msft,d,logc=piece
            y.append(-0.5*msft)
            x[j] = np.concatenate((d[1:],logc))
        log_posterior_ens.append(y)
        ensemble_per_state.append(x)
        
    nstates = len(log_posterior_ens)
    nens = len(chain)
    return ensemble_per_state, log_posterior_ens, nstates, ndims, nens

def read_aem_ensembles_thicknesses(): # not yet implemented
    log_posterior_ens,ensemble_per_state,ndims = [],[],[]
    for i in range(10):
        chain=read_chain('lyr{:02d}/chain.out'.format(i+1))
        msft,n,d,logc=chain[0]
        ndims.append(len(np.concatenate((d,logc)))-1)
        y=[]
        x = np.zeros((len(chain),ndims[i]))
        for j,piece in enumerate(chain):
            msft,n,d,logc=piece
            y.append(msft)
            x[j] = np.concatenate((d[1:],logc))
        log_posterior_ens.append(y)
        ensemble_per_state.append(x)
        
    nstates = len(log_posterior_ens)
    nens = len(chain)
    return ensemble_per_state, log_posterior_ens, nstates, ndims, nens

def get_names(ndim,state,verbose=False):
    ncond = ndim-state
    ndepths = state
    var_names = []
    d_names = ["depth {}".format(i+1) for i in range(ndepths)]
    c_names = ["log cond {}".format(i+1) for i in range(ncond)]
    var_names = []
    for x in range(state):
        var_names.append(d_names[x])
    for x in range(state):
        var_names.append(c_names[x])
    var_names.append(c_names[-1])
    if(verbose): print(var_names)
    return var_names

def create_InferenceData_object(ensemble,log_posterior,log_likelihood=None,variable_names=None):
    nstates = len(ensemble)
    azobjs = []
    for state in range(nstates):
        ndim = ensemble[state].shape[-1]
        samples = ensemble[state]
        #if(variable_names is None):
        #var_names = ["param{}".format(i) for i in range(ndim)]
        #else:
        var_names = get_names(ndim,state)
        if(samples.ndim==2):
            samples_dict = {name: samples[:, i] for i, name in enumerate(var_names)}
        else:
            samples_dict = {name: samples[:,:, i] for i, name in enumerate(var_names)}
            
        log_likelihood_dict = {}
        if(log_likelihood is not None): log_likelihood_dict = {"log_likelihood": log_likelihood[state]}
        log_posterior_dict = {"log_posterior": log_posterior[state]}

        # convert to arviz.InferenceData
        azobjs.append(az.from_dict(
                             posterior=samples_dict,
                             sample_stats=log_posterior_dict,
                             log_likelihood=log_likelihood_dict
                            ))
    return azobjs

def create_InferenceData_object_per_state(ensemble,log_posterior,ndim,state,log_likelihood=None):
    samples = ensemble
    var_names = get_names(ndim,state)
    if(samples.ndim==2):
        samples_dict = {name: samples[:, i] for i, name in enumerate(var_names)}
    else:
        samples_dict = {name: samples[:,:, i] for i, name in enumerate(var_names)}
            
    log_likelihood_dict = {}
    if(log_likelihood is not None): log_likelihood_dict = {"log_likelihood": log_likelihood[state]}
    log_posterior_dict = {"log_posterior": log_posterior[state]}

    return az.from_dict(posterior=samples_dict, # convert to arviz.InferenceData
                        sample_stats=log_posterior_dict,
                        log_likelihood=log_likelihood_dict)

def plot_ensembles(azobj,params=None,truths=None,figsize=None,traceplot=True,filename=None,label_kwargs=None):

    fig,v,t = cornerplot(azobj,truths=truths,params=params,returnnames=True,figsize=figsize,label_kwargs=label_kwargs)
    
    # traceplot
    if(traceplot):
        az.style.use("arviz-doc")
        az.plot_trace(azobj, var_names=v,figsize=fig.get_size_inches())
    if(filename is not None): plt.savefig(filename)
    plt.show()

def cornerplot(arviz_data,title=None,truths=None,params=None,plotall=False,returnnames=False,figsize=None,label_kwargs=None):
    contour_kwargs = {"linewidths" : 0.5}
    data_kwargs = {"color" : "darkblue"}
    data_kwargs = {"color" : "slateblue"}
    #label_kwargs = {"fontsize" : fontsize}
    allvars = list(arviz_data.posterior.keys())
    if(params is None):
        var_names = allvars
    else:
        var_names = [allvars[i] for i in params]  # construct list of variable names to plot
    if(plotall):
        pass
    elif(truths is not None): 
        truths = {var_names[i]: truths[params[i]] for i in range(len(params))} # construct dictionary of truth values for plotting
            
    if(figsize is not None):
        fig = plt.figure(figsize=(10,10))
        corner.corner(
            arviz_data, 
            truths=truths,
            title=title,
            fig=fig,
            var_names=var_names,
            bins=40,hist_bin_factor=2,smooth=True,contour_kwargs=contour_kwargs,data_kwargs=data_kwargs,label_kwargs=label_kwargs
            );  
    else:
        fig = corner.corner(
        arviz_data, 
        truths=truths,
        title=title,
        var_names=var_names,
        bins=40,hist_bin_factor=2,smooth=True,contour_kwargs=contour_kwargs,data_kwargs=data_kwargs,label_kwargs=label_kwargs
        );  
    
    if(returnnames): return fig,var_names,truths
    return fig

def plot_pseudo_samples(pseudo_prior_function,state,ndims,ns=10000,truths=None,params=None,figsize=None,filename=None,label_kwargs=None):
    
    s = np.zeros((1,ns,ndims[state]))
    lp = np.zeros(ns)
    for i in range(ns):
        lp[i],s[0,i,:] = pseudo_prior_function(None,state,returndeviate=True)
        
    azobj = create_InferenceData_object_per_state(s,lp,ndims[state],state) # create list of arviz objetcs to help with plotting
    plot_ensembles(azobj,params=params,truths=truths,figsize=figsize,traceplot=False,filename=filename,label_kwargs=label_kwargs)
    
def ens_diagnostics(iss,elapsed_time,thin=15,discard=0):
    print('\n Algorithm type                                      :', iss.alg)
    print(' Number of walkers                                   :', iss.nwalkers)
    print(' Number of steps                                     :', iss.nsteps)
    #print(' Acceptance rates for walkers within states:  \n',*iss.accept_within_per_walker,'\n')
    #print(' Acceptance rates for walkers between states: \n',*iss.accept_between_per_walker,'\n')
    #print(' Average % acceptance rate for within states         :',np.round(istomo_ens.accept_within,2))
    print(' Average % acceptance rate for between states        :',np.round(iss.accept_between,2))
    # extract trans-D samples and chains
    #discard = 0                  # chain burnin
    #thin = 15                    # chain thinning
    chain,states_chain = iss.get_visits_to_states(discard=discard,thin=thin,normalize=True,walker_average='median',return_samples=True)
    print(' Auto correlation time for between state sampling    :',np.round(iss.autocorr_time_for_between_state_jumps,5))
    print(' Total number of state changes for all walkers       :',iss.total_state_changes)
    print(' Estimated relative evidences ens                    :', *np.round((iss.relative_marginal_likelihoods),5))
    print(' Elapsed time                                        :', np.round(elapsed_time,2),'s \n')
    return chain,states_chain

# set proposals only between neighbouring states (kneighbours is the maximum diagonal to allow a transition)
def setproposalweights(nstates,kneighbours):
    A = np.ones((nstates,nstates))
    A[np.diag_indices(nstates)] = 0.
    if(kneighbours==-1): return A # set transition probabilities for all states equally
    A[np.triu_indices_from(A, k=kneighbours+1)] = 0.
    A[np.tril_indices_from(A, k=-(kneighbours+1))] = 0.
    return A # set transition probabilities between selected states only