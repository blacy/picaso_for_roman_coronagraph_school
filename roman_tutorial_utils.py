import numpy as np
import pandas as pd
import warnings
import os
import glob
from astropy.io import fits
from picaso import justdoit as jdi

# ── Approximate Roman-CGI band definitions ────────────────────────────
# Band 1: photometric,  9.8% fractional bandwidth centred at 0.5738 µm
B1_CTR = 0.5738;  B1_MIN, B1_MAX = B1_CTR * (1.0-0.098/2.0), B1_CTR * (1.0+0.098/2.0)
# Band 3: spectroscopic, 16.8% fractional bandwidth centred at 0.7293 µm, R=50
B3_CTR = 0.7293;  B3_MIN, B3_MAX = B3_CTR * (1.0-0.168/2.0), B3_CTR * (1.0+0.168/2.0);  B3_R = 50
# Band 4: photometric,  11.7% fractional bandwidth centred at 0.8255 µm
B4_CTR = 0.8255;  B4_MIN, B4_MAX = B4_CTR * (1.0-0.117/2.0), B4_CTR * (1.0+0.117/2.0)
# ──────────────────────────────────────────────────────────────────────

_PT_DIR  = os.path.join(os.getenv('picaso_refdata'),'roman_school', 'Batalha2018', 'jfort_pt')
_CLD_DIR = os.path.join(os.getenv('picaso_refdata'),'roman_school', 'Batalha2018', 'jfort_cld')

_VALID_MH   = np.array([0.0, 0.5, 1.0, 1.5, 1.7, 2.0])
_VALID_SEP  = np.array([0.5, 0.6, 0.7, 0.85, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
_VALID_FSED = [(0.01,'0.01'), (0.03,'0.03'), (0.1,'0.1'), (0.3,'0.3'),
               (1,'1'), (3,'3'), (6,'6')]
_FSED_VALS  = np.array([f for f, _ in _VALID_FSED])
_FSED_STRS  = [s for _, s in _VALID_FSED]

_SOLAR_CO     = 0.55   # absolute C/O, Asplund et al. 2009
_VISSCHER_FEH = np.array([-1.5, -1.0, -0.7, -0.5, 0.0, 0.5, 1.0, 1.5])

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


def load_ewi_data(data_dir, exclude=[]):
    """
    Load EWI MOSAIC FITS files for any directly-imaged companion and
    return processed arrays ready for grid fitting.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing *MOSAIC*.fits files
        (e.g. os.path.join('Data', 'ewi-data', 'BetaPicb', 'ForMated')).
    exclude : list of strings or an empty list
        Instrument names to drop before processing.  Useful for skipping
        very-high-resolution datasets that are not needed for a coarse
        grid fit (e.g. exclude=('SINFONI-K4000',)).
        Default None keeps all instruments.

    Returns
    -------
    two dicts with keys
        all_spec   : dictionary of per-instrument dicts (except any excluded by kwarg), 
                     each with keys: wav, flx, err, res, wno
        all_phot   : dictionary of per-photometric filter dicts (except any excluded by kwarg), 
                     each with keys: wav, flx, err, res, wav_min, wav_max
    """
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

    # --- Per-instrument cleanup and derived fields ---
    # For each instrument dict we:
    #   1. Drop points with err <= 0
    #   2. Compute is_phot / is_spec
    #   3. Look up the photometric bandpass; drop phot points with no entry
    #   4. Attach wno_spec, phot_idx, phot_bandpasses
    # All per-instrument keys (is_phot, is_spec, wno_spec, phot_idx,
    # phot_bandpasses) are then available directly on each selected_data entry.
    for d in all_data:
        # 1. Drop bad-error points
        good = d['err'] > 0
        if not np.all(good):
            for key in ('wav', 'flx', 'err', 'res', 'ins_arr'):
                d[key] = d[key][good]

    # let's separate out spectral data from photometric data
    # using the flag that res is set to 0 for photometric data points
    # we'll also exlude any instruments/filters specified by user
    all_spec = {}
    all_phot = {}
    for j in range(len(all_data)):
        ins_dict = all_data[j]
        spec_ids = []
        for k in range(len(ins_dict['res'])):
            if ins_dict['res'][k] == 0:
                # add it to the photometry dictionary
                if ins_dict['ins_arr'][k] not in exclude:
                    all_phot[ins_dict['ins_arr'][k]] = {}
                    for key in ['wav','flx','err']:
                        all_phot[ins_dict['ins_arr'][k]][key] = ins_dict[key][k]
                        all_phot[ins_dict['ins_arr'][k]]['main_ins'] = ins_dict['ins']
                    wlmin, wlmax = EWI_FILTER_BANDPASS[ins_dict['ins_arr'][k]]
                    all_phot[ins_dict['ins_arr'][k]]['wav_min'] = wlmin
                    all_phot[ins_dict['ins_arr'][k]]['wav_max'] = wlmax
                else: pass
            else: 
                spec_ids.append(k) 
        # add spectral data to the spectral dictionary
        # double check you are not over-writing anything
        if len(spec_ids) > 0:
            spec_mask = np.array(spec_ids)
            if ins_dict['ins'] not in exclude:
                ins_key = ins_dict['ins']
                if ins_key in list(all_spec.keys()):
                    count = list(all_spec.keys()).count(ins_key)
                    visit_tag = '_visit_%i'%(count+1)
                    ins_key+= visit_tag
                all_spec[ins_key] = {}
                for key in ['wav','flx','err','res']:
                    all_spec[ins_key][key] = ins_dict[key][spec_mask]
                all_spec[ins_key]['wno'] = 1e4/all_spec[ins_key]['wav'] 

    return all_spec, all_phot


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

def match_picaso_spec_to_ewi_data(picaso_spec, ewi_spec, ewi_phot):
    """
     Parameters
    ----------   
    picaso_spec - a dictionary of spectral output in the form returned by 
                  PICASO's .spectrum  class function. This just needs to include
                  a key 'wavenumber' in ascending order and a key 'thermal'
    ewi_spec - a dictionary of dictionaries of spectroscopic data in the form 
               returned by load_ewi_data
    ewi_phot - a dictionary of dictionaries of photometric data in the form 
               returned by load_ewi_data

    Returns
    -------
    C_num  - the numerator for computing C_model
    C_den  - the denominator for computing C_model
    C_model - the scaling factor C which will produce the smallest chi^2 
              value between data and model spectrum
    model_flux - the picaso_spec flux density binned to match each data point where 
                 we create on flat vector by looping through all of the entries 
                 of ewi_spec and then ewi_phot and compile them into a 1d array
                 NOT sorted by wavelenght or wavenumber
    
    """
    C_num, C_den = 0.0, 0.0
    model_flux = np.array([])
    wno_model, fl_model = picaso_spec['wavenumber'], picaso_spec['thermal']* 1e-7

    for d in [ewi_spec[key] for key in list(ewi_spec.keys())]:
        model_d = np.empty(len(d['wav']))
        # we can use picaso's mean_regrid to match observations, this time
        # passing the newx kwarg
        # note that mean_regrid needs newx to be in ascending wavenumber order
        _, fl_spec = jdi.mean_regrid(wno_model, 
                                     fl_model, 
                                     newx=d['wno'][::-1]) 
        model_d = fl_spec[::-1] # but data is in descending wno order, so we'll match that
        C_num += np.sum(d['flx'] * model_d / d['err']**2)
        C_den += np.sum(model_d**2 / d['err']**2)
        model_flux = np.concatenate((model_flux,model_d))
    
    for d in [ewi_phot[key] for key in list(ewi_phot.keys())]:
        lmin = d['wav_min']
        lmax = d['wav_max']
        bp_mask = (1e4/wno_model >= lmin) & (1e4/wno_model <= lmax)
        model_d = np.mean(fl_model[bp_mask])
        C_num += np.sum(d['flx'] * model_d / d['err']**2)
        C_den += np.sum(model_d**2 / d['err']**2)
        model_flux = np.concatenate((model_flux,np.array([model_d])))
    
    C_model = C_num / C_den

    return C_num, C_den, C_model, model_flux