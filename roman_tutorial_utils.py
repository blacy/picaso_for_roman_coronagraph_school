import numpy as np
import pandas as pd
import warnings
import os
from picaso import justdoit as jdi 

# ── Roman-CGI band definitions ───────────────────────────────────────
# Band 1: photometric,  9.8% fractional bandwidth centred at 0.5738 µm
B1_CTR = 0.5738;  B1_MIN, B1_MAX = B1_CTR * (1.0-0.098/2.0), B1_CTR * (1.0+0.098/2.0)
# Band 3: spectroscopic, 16.8% fractional bandwidth centred at 0.7293 µm, R=50
B3_CTR = 0.7293;  B3_MIN, B3_MAX = B3_CTR * (1.0-0.168/2.0), B3_CTR * (1.0+0.168/2.0);  B3_R = 50
# Band 4: photometric,  11.7% fractional bandwidth centred at 0.8255 µm
B4_CTR = 0.8255;  B4_MIN, B4_MAX = B4_CTR * (1.0-0.117/2.0), B4_CTR * (1.0+0.117/2.0)
# ──────────────────────────────────────────────────────────────────────

def format_roman_cgi_spectrum(axis,ymin,ymax,
                              ylabel='F$_{p}$/F$_{*}$',
                              xlim=(0.4,1.0)):
    """
    lightly shades the wavelength ranges of Roman-CGI, sets axis
    limits, and labels x axis with matplotlib

    Parameters
    ---------- 
          axis - a matplotlib axis object
          ymin - minimum y value in same units as figure
          ymax - maximum y value in same units as figure
          ylabel - label for the y axis
          xlim - lower and upper bounds for the x axis
    Returns
    -------
          nothing
    """
    axis.axvspan(B1_MIN, B1_MAX, alpha=0.12, color='royalblue')
    axis.axvspan(B3_MIN, B3_MAX, alpha=0.12, color='darkorange')
    axis.axvspan(B4_MIN, B4_MAX, alpha=0.12, color='forestgreen')
    
    axis.set_xlabel('Wavelength, $\\mu$m')
    axis.set_ylabel(ylabel)
    axis.set_ylim(ymin,ymax)
    axis.set_xlim(xlim[0],xlim[1])

def mimic_roman_cgi_obs(spec_result_dict,spec_key='fpfs_total'):
    """
    takes in a higher resolution picaso model and
    returns the mean flux density for photometric bands
    band 3 R~50 spectra

    Parameters
    ----------
          spec_result_dict - a dictionary output by the 
                             .spectrum command
          spec_key - the key for which entry of 
                     spec_result_dict you want to use
    Returns
    -------
          tuples for each of bands 1, 3 and 4 which have
          wavelength as the first entry and appropriately
          binned and resampled spectral quantities as second entry
    """
    wno, fpfs = spec_result_dict['wavenumber'], spec_result_dict[spec_key]
    
    mask1 = (1e4/wno >= B1_MIN) & (1e4/wno <= B1_MAX)
    b1 = (B1_CTR,np.mean(fpfs[mask1]))

    wno_R50, fpfs_R50 = jdi.mean_regrid(wno,fpfs, R=50) 
    mask3 = (1e4/wno_R50>B3_MIN) & (1e4/wno_R50<B3_MAX)    
    b3 = (1e4/wno_R50[mask3],fpfs_R50[mask3])
    
    mask4 = (1e4/wno >= B4_MIN) & (1e4/wno <= B4_MAX)
    b4 = (B4_CTR,np.mean(fpfs[mask4]))
    
    return b1, b3, b4

def plot_roman_cgi_obs(ax,spec_result_dict,spec_key='fpfs_total',color='k'):
    """
    convenience function for plotting Roman-CGI sampling of 
    a higher resolution spectrum with matplotlib

    Parameters
    ----------
          ax - a matplotlib axis object 
          spec_result_dict - a dictionary output by the 
                             .spectrum command 
          spec_key - the key for which entry of 
                     spec_result_dict you want to use
        
    Returns
    -------
          nothing
    """
    b1,b3,b4 = mimic_roman_cgi_obs(spec_result_dict,spec_key=spec_key)
    ax.plot(b1[0], b1[1], marker='D',color=color)
    ax.plot(b3[0], b3[1], linewidth=1.5, marker='d',color=color)
    ax.plot(b4[0], b4[1], marker='D',color=color)


_PT_DIR  = os.path.join('Data', 'ColorColorGrid', 'jfort_pt')
_CLD_DIR = os.path.join('Data', 'ColorColorGrid', 'jfort_cld')

_VALID_MH   = np.array([0.0, 0.5, 1.0, 1.5, 1.7, 2.0])
_VALID_SEP  = np.array([0.5, 0.6, 0.7, 0.85, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
_VALID_FSED = [(0.01,'0.01'), (0.03,'0.03'), (0.1,'0.1'), (0.3,'0.3'),
               (1,'1'), (3,'3'), (6,'6')]
_FSED_VALS  = np.array([f for f, _ in _VALID_FSED])
_FSED_STRS  = [s for _, s in _VALID_FSED]

_SOLAR_CO     = 0.55   # absolute C/O, Asplund et al. 2009
_VISSCHER_FEH = np.array([-1.5, -1.0, -0.7, -0.5, 0.0, 0.5, 1.0, 1.5])


def read_batalha_profile(metallicity, separation, fsed=1,
                       pt_dir=_PT_DIR, cld_dir=_CLD_DIR):
    """
    Read a batalha irradiated-Jupiter T-P profile (with Visscher equilibrium chemistry)
    and the corresponding cloud profile, snapping each parameter to the nearest grid point.

    Parameters
    ----------
    metallicity : float
        Log metallicity [M/H] in dex relative to solar.
        Snapped to nearest of: 0.0, 0.5, 1.0, 1.5, 1.7, 2.0
        Chemistry uses Visscher 2121 grid (max [Fe/H] = 1.5; 1.7 and 2.0 snap to 1.5).
    separation : float
        Planet-star separation in AU.
        Snapped to nearest of: 0.5, 0.6, 0.7, 0.85, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0
    fsed : float
        Sedimentation efficiency. Snapped (in log space) to nearest of:
        0.01, 0.03, 0.1, 0.3, 1, 3, 6
    pt_dir : str
        Path to the jfort_pt directory.
    cld_dir : str
        Path to the jfort_cld directory.

    Returns
    -------
    pt_df : pandas.DataFrame
        T-P profile with columns 'pressure' (bars), 'temperature' (K),
        plus one column per molecule (volume mixing ratios) from Visscher
        equilibrium chemistry at solar C/O.
    cld_df : pandas.DataFrame
        Cloud profile with columns 'opd', 'w0', 'g0'.
        Ready for use with case.clouds(df=cld_df).
    """
    # --- Snap all three parameters ---
    mh_snap   = _VALID_MH[np.argmin(np.abs(_VALID_MH  - metallicity))]
    sep_snap  = _VALID_SEP[np.argmin(np.abs(_VALID_SEP - separation))]
    fsed_idx  = np.argmin(np.abs(np.log10(_FSED_VALS) - np.log10(fsed)))
    fsed_snap = _FSED_VALS[fsed_idx]
    fsed_str  = _FSED_STRS[fsed_idx]

    snapped = []
    if mh_snap   != metallicity : snapped.append(f'metallicity {metallicity} → {mh_snap}')
    if sep_snap  != separation  : snapped.append(f'separation {separation} → {sep_snap} AU')
    if fsed_snap != fsed        : snapped.append(f'fsed {fsed} → {fsed_snap}')
    # Warn if chemistry grid caps metallicity
    chem_feh = _VISSCHER_FEH[np.argmin(np.abs(_VISSCHER_FEH - mh_snap))]
    if chem_feh != mh_snap:
        snapped.append(f'chemistry [Fe/H] {mh_snap} → {chem_feh} (Visscher grid max)')
    if snapped:
        print(f"Snapped to nearest grid point: {', '.join(snapped)}")

    mh_str  = str(float(mh_snap))
    sep_str = str(float(sep_snap))

    # --- T-P profile ---
    pt_path = os.path.join(pt_dir, f'm{mh_str}',
                           f'g25_t150_m{mh_str}_d{sep_str}.pt')
    pt_df = pd.read_csv(pt_path, sep=r'\s+', skiprows=1, header=None,
                        usecols=[1, 2], names=['pressure', 'temperature'])

    # --- Add Visscher equilibrium chemistry via PICASO's interpolation ---
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tmp = jdi.inputs()
        tmp.atmosphere(df=pt_df[['pressure', 'temperature']].copy(),
                       chem_method='visscher',
                       mh=float(10**mh_snap),
                       cto_absolute=_SOLAR_CO)
        tmp.chemistry_handler()
    pt_df = tmp.inputs['atmosphere']['profile']

    # --- Cloud profile ---
    cld_path = os.path.join(cld_dir, f'm{mh_str}', f'd{sep_str}',
                            f'm{mh_str}x_rfacv0.5-nc_tint150-f{fsed_str}-d{sep_str}.cld')
    cld_df = pd.read_csv(cld_path, sep=r'\s+', header=None,
                         usecols=[2, 3, 4], names=['opd', 'g0', 'w0'])

    return pt_df, cld_df
