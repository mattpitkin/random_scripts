"""
An attempt to use Enterprise and enterprise_extensions to fit the frequency and
frequency derivative parameters of a pulsar.

This is based on the example given in the Google Colab notebook here https://colab.research.google.com/drive/11aRVepxn_whRm_JWCbgL_sVqn1hjo9Ik#scrollTo=YG3yG0qY8cor (see the "Now, the easy way to do all of this") and using
the data from https://github.com/AaronDJohnson/wn_sp_tutorial.git (which also contains the above notebook).
"""

from enterprise.pulsar import Pulsar
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.sampler import setup_sampler

import numpy as np

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

# use the tim files from wn_sp_tutorial - tim files within the -pta flag
# do not seem to work with model_singlepsr_noise
parfile = "/home/matt/repositories/wn_sp_tutorial/data/par/J0030+0451_NANOGrav_12yv3.gls.par"
timfile = "/home/matt/repositories/wn_sp_tutorial/data/tim/J0030+0451_NANOGrav_12yv3.tim"

psr = Pulsar(parfile, timfile, drop_t2pulsar=False)

print(type(psr))
print(psr.flags)

pta = model_singlepsr_noise(
    psr,
    tmparam_list=["F0", "F1"],  # vary F0 and F1
    tm_linear=False,  # set to False so F0 and F1 will be varied
    tm_var=True,  # set to True so F0 and F1 will be varied
    components=10,  # make smaller than default to try and speed things up
    tm_marg=False,
    red_var=False,  # don't estimate red noise to speed things up
)

print(pta.params)


outdir = "test/"

sampler = setup_sampler(pta, outdir=outdir, resume=False)

N = 5000
x0 = np.hstack(p.sample() for p in pta.params)

print(x0)
print(pta.param_names)

# currently sampler does not seem to vary F0 and F1 params, not sure why!
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, )

