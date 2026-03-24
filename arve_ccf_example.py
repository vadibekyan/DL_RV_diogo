"""
ARVE CCF Example.
"""

#%%
### PACKAGES

import arve
from   glob              import glob
import matplotlib.pyplot as     plt
import numpy             as     np

#%%
### ARVE CCF

# Files
files = sorted(glob("../Data/Input/*.csv"))
Nfile = len(files)

# Times
time_val = np.arange(Nfile)

# Initiate ARVE object
example = arve.ARVE()

# Specify target and fetch parameters from SIMBAD
example.star.target = "Sun"
example.star.get_stellar_parameters()

# Or, specify stellar parameters manually
example.star.target = "Sun"
example.star.stellar_parameters = {
'sptype'  : 'G2',
'vrad_sys': 0.0,
'berv_max': 1.0,
'Teff'    : 5770,
'logg'    : 4.4,
'Fe_H'    : 0.0,
'M'       : 1.0,
'R'       : 1.0,
'vsini'   : 1.63,
'vmic'    : 0.85,
'vmac'    : 3.98
}

# Add files and specifications
# If stored in individual CSV files, the spectra must contain the following columns/keywords: "wave_val", "flux_val", "flux_err".
example.data.add_data(time_val=time_val, files=files, format="s1d", extension="csv", medium="vac", resolution=100000, same_wave_grid=True)

# Add custom CCF mask
# The mask must be a CSV file and contain the columns: "wave", "wave_l", "wave_u" ("wave_l" and "wave_u" can have the same values as "wave").
# It can optionally contain a weight column whose name must match what is used by the "weight_name" argument in the compute_vrad_ccf() function.
example.data.get_aux_data(mask_path="../Data/Output/mask.csv", mask_medium="vac")

# Compute CCFs and CCF RVs
example.data.compute_vrad_ccf(exclude_tellurics=False, vrad_grid=[-20,20,1], ccf_err_scale=True, weight_name="weight")

# Extract CCFs
ccf_vrad = example.data.ccf["ccf_vrad"]
ccf_val  = example.data.ccf["ccf_val" ][:,-1,:]
ccf_err  = example.data.ccf["ccf_err" ][:,-1,:]

# Extract CCF RVs
time_val = example.data.time["time_val"]
vrad_val = example.data.vrad["vrad_val"]
vrad_err = example.data.vrad["vrad_err"]

#%%
### PLOT

# Plot CCFs
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(ccf_vrad, np.median(ccf_val, axis=0), "-k")
axs[0].set_ylabel("CCF")
for i in range(Nfile):
    axs[1].plot(ccf_vrad, ccf_val[i]-np.median(ccf_val, axis=0), "-")
axs[1].set_xlabel("RV [km/s]")
axs[1].set_ylabel("$\Delta$CCF")
fig.align_ylabels()
plt.tight_layout()
plt.show()

# Plot RVs
plt.figure()
plt.errorbar(time_val, vrad_val, vrad_err, fmt=".", color="k", ecolor="r")
plt.xlabel("Time")
plt.ylabel("RV [km/s]")
plt.tight_layout()
plt.show()

#%%