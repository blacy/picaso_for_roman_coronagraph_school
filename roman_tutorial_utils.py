import numpy as np
import pandas as pd
import warnings
import os
import glob
from astropy.io import fits
from picaso import justdoit as jdi

# ── Roman-CGI band definitions ───────────────────────────────────────
# Band 1: photometric,  9.8% fractional bandwidth centred at 0.5738 µm
B1_CTR = 0.5738;  B1_MIN, B1_MAX = B1_CTR * (1.0-0.098/2.0), B1_CTR * (1.0+0.098/2.0)
# Band 3: spectroscopic, 16.8% fractional bandwidth centred at 0.7293 µm, R=50
B3_CTR = 0.7293;  B3_MIN, B3_MAX = B3_CTR * (1.0-0.168/2.0), B3_CTR * (1.0+0.168/2.0);  B3_R = 50
# Band 4: photometric,  11.7% fractional bandwidth centred at 0.8255 µm
B4_CTR = 0.8255;  B4_MIN, B4_MAX = B4_CTR * (1.0-0.117/2.0), B4_CTR * (1.0+0.117/2.0)
# ──────────────────────────────────────────────────────────────────────

# ── EWI photometric filter bandpasses ────────────────────────────────────────
# Keys match the instrument abbreviations found in the EWI FITS 'INS' column.
# Values are (lambda_min, lambda_max) in µm.
# Used internally by load_ewi_data(); pass as filter_bandpass= to override.
EWI_FILTER_BANDPASS = {
    # Magellan / MagAO
    'MagYs'   : (0.960, 1.070),
    # VLT / NACO
    'NaCoJ'   : (1.170, 1.330),
    'NaCoH'   : (1.500, 1.800),
    'NaCoKs'  : (2.000, 2.300),
    'NaCoLp'  : (3.430, 4.130),
    'NaCo405' : (4.042, 4.062),
    'NaCoMp'  : (4.550, 5.000),
    # MagAO / Clio
    'Clio31'  : (3.052, 3.152),
    'Clio33'  : (3.260, 3.430),
    'ClioLp'  : (3.430, 4.130),
    'ClioMp'  : (4.550, 5.000),
    # Keck / NIRC2
    'NIRC2J'  : (1.170, 1.330),
    'NIRC2H'  : (1.490, 1.780),
    'NIRC2Ks' : (1.990, 2.310),
    'NIRC2Lp' : (3.430, 4.130),
    'NIRC2Ms' : (4.550, 5.140),
    # Gemini / GPI (photometric channels)
    'GPIY'    : (0.950, 1.140),
    'GPIJ'    : (1.120, 1.350),
    'GPIH'    : (1.490, 1.800),
    'GPIK'    : (1.900, 2.400),
    # JWST / NIRCam broadband
    'F115W'   : (1.013, 1.282),
    'F150W'   : (1.331, 1.668),
    'F200W'   : (1.755, 2.227),
    'F356W'   : (3.140, 3.980),
    'F444W'   : (3.880, 4.986),
    # JWST / MIRI photometric filters
    'F560W'   : (5.005, 6.303),
    'F770W'   : (6.636, 8.675),
}
# ──────────────────────────────────────────────────────────────────────────────

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
    b3 = (1e4/wno_R50[mask3][::-1],fpfs_R50[mask3][::-1]) # put this in ascending order...
    
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


def load_ewi_data(data_dir, exclude=None, filter_bandpass=None):
    """
    Load EWI MOSAIC FITS files for any directly-imaged companion and
    return processed arrays ready for grid fitting.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing *MOSAIC*.fits files
        (e.g. os.path.join('Data', 'ewi-data', 'BetaPicb', 'ForMated')).
    exclude : sequence of str or None
        Instrument names to drop before processing.  Useful for skipping
        very-high-resolution datasets that are not needed for a coarse
        grid fit (e.g. exclude=('SINFONI-K4000',)).
        Default None keeps all instruments.
    filter_bandpass : dict or None
        Map of instrument name → (lmin_µm, lmax_µm) used to average the
        model spectrum over each photometric bandpass.  If None (default),
        uses the built-in EWI_FILTER_BANDPASS table.  Pass a dict to add
        or override entries for non-standard instruments.  Photometric
        points whose instrument name is not found in the table are dropped
        with a warning.

    Returns
    -------
    dict with keys
        all_data        : list of per-instrument dicts (all instruments,
                          including any that were excluded), each with
                          keys wav, flx, err, res, ins, ins_arr.
        plot_data       : same structure as all_data but with excluded
                          instruments removed — use this for SED plots
                          so excluded data is not displayed.
        wl              : ndarray — wavelength in µm, sorted ascending,
                          points with err <= 0 removed
        flx             : ndarray — flux in W m⁻² µm⁻¹
        err             : ndarray — flux uncertainty in W m⁻² µm⁻¹
        res             : ndarray — spectral resolution flag (0 = photometric)
        ins             : ndarray of str — instrument name per point
        is_phot         : bool ndarray — True where res == 0
        is_spec         : bool ndarray — True where res > 0
        wno_spec        : ndarray — wavenumber (cm⁻¹, increasing order) for
                          spectroscopic points; pass as newx to mean_regrid
        phot_bandpasses : list of (lmin, lmax) tuples, one per phot point
        phot_idx        : int ndarray — indices of photometric points in the
                          main sorted arrays
    """
    bp = EWI_FILTER_BANDPASS if filter_bandpass is None else filter_bandpass

    # --- Load all MOSAIC files ---
    mosaic_files = sorted(glob.glob(os.path.join(data_dir, '*MOSAIC*.fits')))
    if not mosaic_files:
        raise FileNotFoundError(f"No *MOSAIC*.fits files found in {data_dir!r}")

    all_data = []
    for fpath in mosaic_files:
        with fits.open(fpath) as hdul:
            tbl = hdul[1].data
            wav = tbl['WAV'].astype(float)
            flx = tbl['FLX'].astype(float)
            err = tbl['ERR'].astype(float)
            res = tbl['RES'].astype(float)
            ins = tbl['INS']
        ins_arr  = np.array([str(s).strip() for s in ins])
        ins_name = ins_arr[0]
        all_data.append(dict(wav=wav, flx=flx, err=err, res=res,
                             ins=ins_name, ins_arr=ins_arr))

    # --- Filter excluded instruments ---
    exclude_set = set(exclude) if exclude is not None else set()
    obs_data = [d for d in all_data if d['ins'] not in exclude_set]

    # --- Concatenate, sort by wavelength, drop bad errors ---
    wl_all  = np.concatenate([d['wav']     for d in obs_data])
    flx_all = np.concatenate([d['flx']     for d in obs_data])
    err_all = np.concatenate([d['err']     for d in obs_data])
    res_all = np.concatenate([d['res']     for d in obs_data])
    ins_all = np.concatenate([d['ins_arr'] for d in obs_data])

    idx = np.argsort(wl_all)
    wl_all, flx_all, err_all, res_all, ins_all = (
        a[idx] for a in [wl_all, flx_all, err_all, res_all, ins_all])

    good = err_all > 0
    wl_obs  = wl_all [good]
    flx_obs = flx_all[good]
    err_obs = err_all[good]
    res_obs = res_all[good]
    ins_obs = ins_all[good]

    # --- Masks ---
    is_phot = res_obs == 0
    is_spec = ~is_phot

    # Spectroscopic wavenumber grid in increasing order (required by mean_regrid)
    wno_spec = (1e4 / wl_obs[is_spec])[::-1]

    # --- Photometric bandpasses ---
    phot_bandpasses = []
    phot_keep       = np.ones(is_phot.sum(), dtype=bool)
    phot_positions  = np.where(is_phot)[0]

    for m, ins in enumerate(ins_obs[is_phot]):
        if ins in bp:
            phot_bandpasses.append(bp[ins])
        else:
            warnings.warn(
                f"Photometric point from instrument '{ins}' has no entry in "
                f"EWI_FILTER_BANDPASS and will be dropped.  "
                f"Pass filter_bandpass={{'{ins}': (lmin, lmax)}} to include it.",
                UserWarning, stacklevel=2)
            phot_keep[m] = False

    # Drop photometric points with no bandpass entry
    if not np.all(phot_keep):
        drop_idx = phot_positions[~phot_keep]
        keep_mask = np.ones(len(wl_obs), dtype=bool)
        keep_mask[drop_idx] = False
        wl_obs  = wl_obs [keep_mask]
        flx_obs = flx_obs[keep_mask]
        err_obs = err_obs[keep_mask]
        res_obs = res_obs[keep_mask]
        ins_obs = ins_obs[keep_mask]
        is_phot = res_obs == 0
        is_spec = ~is_phot
        wno_spec = (1e4 / wl_obs[is_spec])[::-1]

    phot_idx = np.where(is_phot)[0]

    return dict(
        all_data        = all_data,
        plot_data       = obs_data,   # filtered list (respects exclude); use for SED plots
        wl              = wl_obs,
        flx             = flx_obs,
        err             = err_obs,
        res             = res_obs,
        ins             = ins_obs,
        is_phot         = is_phot,
        is_spec         = is_spec,
        wno_spec        = wno_spec,
        phot_bandpasses = phot_bandpasses,
        phot_idx        = phot_idx,
    )


def plot_chi2_grid(ax, z_data, x_vals, y_vals,
                   xlabel, ylabel, title,
                   truth_x, truth_y,
                   chi2_min=None, delta=5.0):
    """
    Plot a 2D chi-squared map as an imshow panel with grid-point tick
    labels, a 1-sigma contour, and a star marking the truth/reference point.

    Parameters
    ----------
    ax : matplotlib Axes
    z_data : 2D array, shape (len(x_vals), len(y_vals))
        Chi-squared values, typically min-projected over a third parameter
        axis (e.g. chi2.min(axis=2)).
    x_vals : array-like
        Parameter values along the x axis (used for tick labels).
    y_vals : array-like
        Parameter values along the y axis (used for tick labels).
    xlabel, ylabel, title : str
    truth_x, truth_y : float
        Reference (truth or published) parameter values, marked with a red star.
    chi2_min : float or None
        Lower bound of the color scale.  If None (default), computed as
        np.nanmin(z_data).  Pass a shared value across panels to keep a
        consistent color scale for direct comparison.
    delta : float
        Color scale range above chi2_min (vmax = chi2_min + delta).
        Default 5.0 (≈ 2σ for a chi-squared distribution).

    Returns
    -------
    None  (modifies ax in-place)
    """
    if chi2_min is None:
        chi2_min = float(np.nanmin(z_data[np.isfinite(z_data)]))

    im = ax.imshow(z_data.T, origin='lower', aspect='auto', cmap='viridis_r',
                   vmin=chi2_min, vmax=chi2_min + delta,
                   extent=[-0.5, len(x_vals) - 0.5, -0.5, len(y_vals) - 0.5])
    ax.figure.colorbar(im, ax=ax, label='min χ²_ν')

    # Contour at Δχ² = 1 (≈ 1σ confidence region)
    try:
        ax.contour(z_data.T, levels=[chi2_min + 1], colors='white',
                   linewidths=1.5,
                   extent=[-0.5, len(x_vals) - 0.5, -0.5, len(y_vals) - 0.5])
    except Exception:
        pass

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Mark truth/reference point at its exact parameter value using interpolated coordinates
    xi = float(np.interp(truth_x, x_vals, np.arange(len(x_vals))))
    yi = float(np.interp(truth_y, y_vals, np.arange(len(y_vals))))
    ax.plot(xi, yi, 'r*', ms=16, zorder=10, label='Reference')
    ax.annotate('lit', (xi, yi), fontsize=8, color='r')
