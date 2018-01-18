"""
Neuroimaging cartesian reconstruction
=====================================

Credit: A Grigis, L Elgueddari, H Carrie

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pISAP, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
from pisap.plugins.mri.reconstruct_3D.utils import imshow3D
from pisap.plugins.mri.reconstruct_3D.utils import prod_over_maps
from pisap.plugins.mri.reconstruct_3D.utils import normalize_samples
from pisap.plugins.mri.reconstruct_3D.utils import function_over_maps
from pisap.plugins.mri.reconstruct_3D.utils import convert_locations_to_mask
from pisap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D
from pisap.plugins.mri.reconstruct_3D.gradient import GradSynthesis3
from pisap.plugins.mri.reconstruct_3D.gradient import GradAnalysis3
from pisap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pisap.plugins.mri.reconstruct_3D.fourier import FFT3
from pisap.plugins.mri.reconstruct_3D.reconstruct import sparse_rec_fista
from pisap.plugins.mri.reconstruct_3D.reconstruct import sparse_rec_condatvu


# Third party import
import numpy as np
import scipy.fftpack as pfft
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load input data
filename = '/neurospin/optimed/LoubnaElGueddari/p_MRI_CSGRE_3D/' \
            '2018-01-15_32ch_ref_nc1000_data/' \
            'meas_MID14_gre_800um_iso_128x128x128_FID24.mat'
workspace = loadmat(filename)
Il = workspace['Il']
Iref = workspace['ref']
Smaps = workspace['Smaps']
imshow3D(Iref, display=False)


samples = loadmat('/neurospin/optimed/clazarus/Data/MRI_rawdata/CSGRE_3D/'
                  '2018-01-15_32ch_ref_nc1000_ddata/'
                  'samples_N256_nc1000_ns3072_ddata.mat')['samples']
samples = normalize_samples(samples)
cartesian_samples = convert_locations_to_mask(samples, [128, 128, 128])

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
kspace_mask = cartesian_samples  # pfft.ifftshift(cartesian_samples)
kspace_data = prod_over_maps(function_over_maps(pfft.fftn, Il), kspace_mask)

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations_3D(kspace_mask)

# Zero order solution
image_rec0 = np.sum(prod_over_maps(function_over_maps(pfft.ifftn, kspace_data),
                    np.conj(Smaps)), 3)
imshow3D(np.abs(image_rec0), display=True)

max_iter = 2

linear_op = pyWavelet3(wavelet_name="sym4",
                       nb_scale=4)

fourier_op = FFT3(samples=kspace_loc, shape=Iref.shape)
gradient_op = GradSynthesis3(data=kspace_data,
                             fourier_op=fourier_op,
                             linear_op=linear_op,
                             Smaps=Smaps)

x_final, transform, cost = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    mu=5e-2,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1,
    get_cost=True)

gradient_op_cd = GradAnalysis3(data=kspace_data,
                               fourier_op=fourier_op,
                               Smaps=Smaps)

x_final, transform = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    std_est=None,
    std_est_method="dual",
    std_thr=2.,
    mu=1e-5,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=2,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)

imshow3D(np.abs(x_final), display=True)
plt.figure()
plt.plot(cost)
plt.show()
