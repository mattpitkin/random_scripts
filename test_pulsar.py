"""
An attempt to use Enterprise and enterprise_extensions to fit the frequency and
frequency derivative parameters of a pulsar.

This is based on the example given in the Google Colab notebook here https://colab.research.google.com/drive/11aRVepxn_whRm_JWCbgL_sVqn1hjo9Ik#scrollTo=YG3yG0qY8cor (see the "Now, the easy way to do all of this") and using
the data from https://github.com/AaronDJohnson/wn_sp_tutorial.git (which also contains the above notebook).

The .tim data for J0030+0451 from the above git repo contains multiple data set, which have different noise properties
that the code will try and fit white noise parameters for. To simplify things I have used a version for which I have
extracted only the L-wide_PUPPI data:

```python
with open("J0030+0451_NANOGrav_12yv3.tim", "r") as fp:
    toalines = fp.readlines()

newfile = "J0030+0451.tim"

with open(newfile, "w") as fp:
   for line in toalines:
       if line[0] in ["C", "M", "F"] or "L-wide_PUPPI" in line:
           fp.write(line)
```
"""

import os

from enterprise.pulsar import Pulsar
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.sampler import setup_sampler

from corner import corner
import numpy as np

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import bilby
from bilby.core.result import read_in_result

# using my version of enterprise_warp https://github.com/mattpitkin/enterprise_warp/tree/tm_params
from enterprise_warp import bilby_warp


# use the tim files from wn_sp_tutorial - tim files within the -pta flag
# do not seem to work with model_singlepsr_noise
parfile = "J0030+0451_NANOGrav_12yv3.gls.par"
timfile = "J0030+0451.tim"

psr = Pulsar(parfile, timfile, drop_t2pulsar=False)

plist = ["F0", "F1"]  # vary F0 and F1

pta = model_singlepsr_noise(
    psr,
    tmparam_list=plist,
    tm_linear=False,  # set to False so F0 and F1 will be varied
    tm_var=True,  # set to True so F0 and F1 will be varied
    components=1,  # make smaller than default to try and speed things up
    tm_marg=False,
    red_var=False,  # don't estimate red noise to speed things up
    white_vary=True,  # estimate white noise (this defaults to True anyway)
)


outdir = "test/"

# set up and run PTMCMC sampler
#sampler = setup_sampler(pta, outdir=outdir, resume=False)

#N = 20000  # number of iterations
#x0 = np.hstack(p.sample() for p in pta.params)  # initial sample points

# run sampler 
#sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

# get parameter indices in output chain file
#i = 0
#paramidx = []
#for p in plist:
#    print(p)
#    for pn in pta.param_names:
#        if f"tmparams_{i}" in pn:
#            print(psr.t2pulsar[p].val)
#            # get parameter name, index in chain file, mean value from par file, stddev from par file
#            paramidx.append((p, i, psr.t2pulsar[p].val, psr.t2pulsar[p].err))
#            i += 1
#            break
            
#print(paramidx)

#samples = np.loadtxt(os.path.join(outdir, "chain_1.txt"))

#nburn = int(samples.shape[0] / 2)

#psamples = []
#for item in paramidx:
#    psamp = samples[nburn:, item[1]] * item[3] + item[2]
#    psamples.append(psamp)

# plot the posteriors on F0 and F1
#fig = corner(np.array(psamples).T)
#fig.savefig("testpost.png", dpi=200)

# run using enterprise_warp to access bilby_mcmc
priors = bilby_warp.get_bilby_prior_dict(pta)
parameters = dict.fromkeys(priors.keys())
likelihood = bilby_warp.PTABilbyLikelihood(pta, parameters)

label = "test_bilby"
#bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir, label=label, sampler="bilby_mcmc", nsamples=1000)

res = read_in_result(os.path.join(outdir, f"{label}_result.json"))

# convert F0 and F1 into true ranges
for i, p in enumerate(["F0", "F1"]):
    res.posterior[f"J0030+0451_timing model_tmparams_{i}"] = psr.t2pulsar[p].val + psr.t2pulsar[p].err * res.posterior[f"J0030+0451_timing model_tmparams_{i}"]

fig = res.plot_corner(filename="test_bilby.png", labels=["EFAC", "ECORR", "EQUAD", "$f_0$ (Hz)", "$\dot{f}$ (Hz/s)"])
