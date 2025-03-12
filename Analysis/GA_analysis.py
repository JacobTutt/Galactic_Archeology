import numpy as np
import pandas as pd
from tqdm import tqdm
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
from dustmaps.sfd import SFDQuery
from dustmaps.config import config
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm
import healpy as hp

def query_gaia_halo_rgb(ra_min, ra_max, dec_min, dec_max, g_max=22.5, parallax_max=1.0, ruwe_max=1.4, chunk_size=0.5, radial_velocity=True, save_path=None):
    """
    Query Gaia DR3 for Red Giant Branch (RGB) stars in the Galactic halo, handling query limits by batching.

    Parameters:
        ra_min (float): Minimum Right Ascension (degrees)
        ra_max (float): Maximum Right Ascension (degrees)
        dec_min (float): Minimum Declination (degrees)
        dec_max (float): Maximum Declination (degrees)
        chunk_size (float, optional): Size of RA/Dec bins (default=0.5 degrees)
        g_max (float, optional): Maximum G-band magnitude (default=22.5)
        parallax_max (float, optional): Maximum parallax to filter distant stars (default=1.0)
        ruwe_max (float, optional): Maximum RUWE value to filter high-quality astrometry (default=1.4)
        radial_velocity (bool, optional): If True, filter only stars with radial velocity measurements.
        save_path (str, optional): Path to save the output CSV file.

    Returns:
        pandas.DataFrame: Merged query results for all chunks (also saved as CSV).
    """

    results_list = []

    # Define RA and Dec steps for batch queries
    ra_steps = np.arange(ra_min, ra_max, chunk_size)
    dec_steps = np.arange(dec_min, dec_max, chunk_size)

    # Total number of queries needed
    total_batches = len(ra_steps) * len(dec_steps)
    print(f"Total queries required: {total_batches}")

    # Progress bar for tracking execution
    with tqdm(total=total_batches, desc="Querying Gaia DR3") as pbar:
        for ra1 in ra_steps:
            for dec1 in dec_steps:
                ra2 = min(ra1 + chunk_size, ra_max)
                dec2 = min(dec1 + chunk_size, dec_max)

                # Apply radial velocity filter only if enabled
                rv_filter = "AND radial_velocity IS NOT NULL" if radial_velocity else ""

                query = f"""
                SELECT source_id, l, b, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                       parallax, parallax_error, ruwe, pmra, pmdec, radial_velocity, mh_gspphot, 
                       teff_gspphot, logg_gspphot
                FROM gaiadr3.gaia_source
                WHERE phot_g_mean_mag BETWEEN 0 AND {g_max}  -- Exclude stars with G < 0 and G > g_max
                AND parallax < {parallax_max}  -- Remove nearby disk stars
                AND ruwe < {ruwe_max}  -- High-quality astrometry
                AND logg_gspphot < 3.0  -- Select Red Giants (exclude subgiants/dwarfs)
                AND teff_gspphot < 4800  -- Remove hotter stars
                {rv_filter}  -- Apply radial velocity filter if enabled
                AND ra BETWEEN {ra1} AND {ra2}
                AND dec BETWEEN {dec1} AND {dec2}
                """

                try:
                    job = Gaia.launch_job(query)
                    chunk_results = job.get_results().to_pandas()

                    if not chunk_results.empty:
                        results_list.append(chunk_results)

                        # Warning if the batch size is too large
                        if len(chunk_results) >= 2000:
                            print(f"Warning: Batch for RA [{ra1}, {ra2}], Dec [{dec1}, {dec2}] returned {len(chunk_results)} stars (≥ 2000). Consider reducing chunk size.")

                except Exception as e:
                    print(f"Query failed for RA [{ra1}, {ra2}], Dec [{dec1}, {dec2}]: {e}")

                pbar.update(1)

    # Combine all results
    full_results = pd.concat(results_list, ignore_index=True)
    print(f"Total stars retrieved: {len(full_results)}")

    if save_path is not None:
        full_results.to_csv(save_path, index=False)
        print(f"Data saved to: {save_path}")

    return full_results


def reddening_correction(gaia_data, dustmaps_dir='dustmaps/'):
    """
    Applies Galactic extinction corrections to Gaia DR3 photometry using the 
    Schlegel, Finkbeiner & Davis (1998) (SFD) dust map and extinction coefficients 
    from Casagrande et al. (2021). Also filters out any rows with missing values in required fields.

    Parameters:
        gaia_data (pd.DataFrame): DataFrame containing Gaia photometric and positional data.
                                  Required columns: 'ra', 'dec', 'phot_g_mean_mag', 
                                                    'phot_bp_mean_mag', 'phot_rp_mean_mag'.
        dustmaps_dir (str, optional): If specified, sets the directory where dustmaps are stored.

    Returns:
        pd.DataFrame: Updated DataFrame with extinction corrections applied and cleaned of missing values.
                      Adds columns: 'dered_G', 'dered_BP', 'dered_RP'.

    Raises:
        ValueError: If any required column is missing from `gaia_data`.
    """
    
    # --------------- Validate Required Columns ---------------
    required_columns = {'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'}
    missing_columns = required_columns - set(gaia_data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in gaia_data: {missing_columns}")

    # --------------- Filter Out Missing Values ---------------
    initial_count = len(gaia_data)
    gaia_data = gaia_data.dropna(subset=required_columns)
    removed_count = initial_count - len(gaia_data)

    if removed_count > 0:
        print(f"{removed_count} rows removed due to missing values in required columns.")

    # --------------- Set Dustmaps Data Directory ---------------
    config['data_dir'] = dustmaps_dir  

    # --------------- Compute Galactic Extinction ---------------
    
    # Convert RA, Dec to SkyCoord
    coords = SkyCoord(ra=gaia_data['ra'].values * u.deg, dec=gaia_data['dec'].values * u.deg, frame="icrs")

    # Query the Schlegel, Finkbeiner & Davis (SFD) dust map
    sfd = SFDQuery()
    # Returns E(B-V) reddening along the line of sight
    ebv = sfd(coords)  

    # Compute extinction coefficients for Gaia bands (Casagrande et al. 2021)
    C = gaia_data['phot_bp_mean_mag'] - gaia_data['phot_rp_mean_mag']
    R_G = 2.609 - 0.475 * C + 0.053 * C**2
    R_BP = 2.998 - 0.140 * C - 0.175 * C**2 + 0.062 * C**3
    R_RP = 1.689 - 0.059 * C

    # Compute dereddened magnitudes
    gaia_data['dered_G'] = gaia_data['phot_g_mean_mag'] - R_G * ebv
    gaia_data['dered_BP'] = gaia_data['phot_bp_mean_mag'] - R_BP * ebv
    gaia_data['dered_RP'] = gaia_data['phot_rp_mean_mag'] - R_RP * ebv

    return gaia_data


### Halo RGB Stars filtering
def halo_rgb_filter(gaia_data, min_br=1, max_br=2.5, min_abs_mag=3, halo_radius_min=15, halo_radius_max=300, pm_min = 5):
    """
    Filters Gaia DR3 data to select Red Giant Branch (RGB) stars in the Galactic halo.

    This function applies multiple selection criteria to extract halo RGB stars:
    1. **Parallax Cut**: Removes stars with negative parallaxes.
    2. **Distance Calculation**: Computes distance (kpc) from parallax.
    3. **Absolute Magnitude Calculation**: Computes Gaia G-band absolute magnitude M_G.
    4. **Galactocentric Transformation**: Converts coordinates to Galactocentric cylindrical frame.
    5. **Color Selection**: Filters stars based on (BP - RP) color range for red giants.
    6. **Luminosity Selection**: Filters by absolute magnitude to exclude dwarfs.
    7. **Galactic Radius Selection**: Filters out disk stars by requiring R_gc > halo_radius.

    Parameters:
        gaia_data (pd.DataFrame): Input DataFrame containing Gaia DR3 data. 
                                  Required columns: 
                                  'ra', 'dec', 'parallax', 'dered_G', 'dered_BP', 'dered_RP'.
        min_br (float, optional): Minimum (BP - RP) color index for RGB selection. Default is 1.0.
        max_br (float, optional): Maximum (BP - RP) color index for RGB selection. Default is 2.5.
        min_abs_mag (float, optional): Minimum absolute magnitude M_G to exclude bright giants. Default is 3.
        halo_radius (float, optional): Minimum Galactocentric radius (kpc) for halo selection. Default is 15.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only halo RGB stars.
                      Adds the following computed columns:
                      - 'dist_kpc': Distance from Sun in kpc.
                      - 'M_G': Absolute G-band magnitude.
                      - 'R_gc': Galactocentric radius (kpc).
                      - 'phi_gc': Galactic azimuthal angle (deg).
                      - 'z_gc': Galactocentric height (kpc).

    Notes:
        - Parallax values must be positive, as negative parallaxes are unphysical.
        - The absolute magnitude calculation follows the standard distance modulus equation:
          M_G = G - 5 log10(d_pc) + 5.
        - Galactocentric coordinates are computed assuming R_0 = 8.1 kpc and z_sun = 25 pc.
        - The color and magnitude cuts are designed to select RGB stars and exclude main-sequence dwarfs.
    """

    # Remove negative parallaxes
    gaia_data = gaia_data[gaia_data["parallax"] > 0].copy()

    # Compute distance (kpc) and absolute magnitude
    gaia_data["dist_kpc"] = 1 / gaia_data["parallax"]  
    gaia_data["M_G"] = gaia_data["dered_G"] - 5 * np.log10(gaia_data["dist_kpc"] * 1000) + 5

    # Compute Proper Motion (Total)
    gaia_data["pm_total"] = np.sqrt(gaia_data["pmra"]**2 + gaia_data["pmdec"]**2)

    # Transform to Galactocentric coordinates
    coords_icrs = SkyCoord(ra=gaia_data['ra'].values * u.deg,  
                           dec=gaia_data['dec'].values * u.deg, 
                           distance=gaia_data['dist_kpc'].values * u.kpc, frame='icrs')
    gc_frame = Galactocentric(galcen_distance=8.1 * u.kpc, z_sun=25 * u.pc)
    coords_gc = coords_icrs.transform_to(gc_frame)
    coords_gc.representation_type = 'cylindrical'

    # Extract Galactocentric distances
    gaia_data["R_gc_cyl"] = coords_gc.rho.to(u.kpc).value
    gaia_data["phi_gc_cyl"] = 180 - coords_gc.phi.to(u.deg).value  # Convert to left-handed system
    gaia_data["z_gc_cyl"] = coords_gc.z.to(u.kpc).value

    # Define filters
    # Based on BP-RP color index 
    mask_bp_rp = (gaia_data['dered_BP'] - gaia_data['dered_RP'] > min_br) & \
                 (gaia_data['dered_BP'] - gaia_data['dered_RP'] < max_br)
    # Based on magnitudes
    mask_abs_mag = gaia_data["M_G"] < min_abs_mag 
    # Based on Galactocentric radius
    total_radius = (gaia_data["R_gc_cyl"]**2 + gaia_data["z_gc_cyl"]**2)**(1/2) # Overall Radius
    mask_halo = (total_radius > halo_radius_min) & (total_radius < halo_radius_max)
    # Proper Motion 
    mask_pm = gaia_data["pm_total"] < pm_min  # Only retain stars with low proper motion
    # Apply combined filter
    mask_total = mask_bp_rp & mask_abs_mag & mask_halo & mask_pm

    # Count stars passing each filter
    num_total = len(gaia_data)
    num_bp_rp = mask_bp_rp.sum()
    num_abs_mag = mask_abs_mag.sum()
    num_halo = mask_halo.sum()
    num_passed = mask_total.sum()

    print(f"Total stars before filtering: {num_total}")
    print(f"Stars passing BP-RP color filter: {num_bp_rp} ({num_bp_rp/num_total:.2%})")
    print(f"Stars passing absolute magnitude filter: {num_abs_mag} ({num_abs_mag/num_total:.2%})")
    print(f"Stars passing halo distance filter: {num_halo} ({num_halo/num_total:.2%})")
    print(f"Stars passing proper motion filter: {mask_pm.sum()} ({mask_pm.sum()/num_total:.2%})")
    print(f"Stars passing all filters: {num_passed} ({num_passed/num_total:.2%})")

    return gaia_data[mask_total]



def plot_sky_density(gaia_data, bins=200, contrast=(5, 95), binning_method="linear", 
                     cmap_density="magma", cmap_rgb="plasma", log_scale=True):
    """
    Creates two visualizations:
    1. A density contrast map of stars in the sky (RA/Dec).
    2. A false-color RGB composite image highlighting Galactic substructures.

    Parameters:
    -----------
    gaia_data : pd.DataFrame
        DataFrame containing 'ra', 'dec', and 'dered_G' magnitudes.
    bins : int, optional
        Number of bins for the 2D histograms (default=200 for higher resolution).
    contrast : tuple, optional
        Percentile range for contrast enhancement (default=(5, 95)).
    binning_method : str, optional
        Method for defining RGB bins ('linear' or 'normal'). Default is 'linear'.
    cmap_density : str, optional
        Colormap for the sky density map (default="magma" for improved contrast).
    cmap_rgb : str, optional
        Colormap to test alternative RGB mappings (default="plasma").
    log_scale : bool, optional
        Whether to use a logarithmic color scale (default=True).

    Returns:
    --------
    None. Displays two high-quality plots.
    """

    ra = gaia_data['ra'].values
    dec = gaia_data['dec'].values
    mag = gaia_data['dered_G'].values

    # --------- Compute RGB Magnitude Bins ---------
    def compute_rgb_bins(mag, method="linear"):
        """
        Compute RGB bins using either 'linear' or 'normal' binning.

        Parameters:
            mag (array-like): Magnitude values.
            method (str): 'linear' for equal-width bins, 'normal' for normal distribution binning.

        Returns:
            list: Three bin edges for R/G/B assignment.
        """
        mag = np.array(mag)
        mag = mag[np.isfinite(mag)]  # Remove NaN values

        if method == "linear":
            return np.linspace(np.min(mag), np.max(mag), 4)  # Three bins

        elif method == "normal":
            mean, std = norm.fit(mag)
            return [np.percentile(mag, 16), np.percentile(mag, 50), np.percentile(mag, 84)]

        else:
            raise ValueError("Invalid method. Choose 'linear' or 'normal'.")

    # Compute RGB magnitude bins
    rgb_bins = compute_rgb_bins(mag, binning_method)

    # --------- Plot Histogram of Magnitude with RGB Bins ---------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    plt.hist(mag, bins=40, color='gray', alpha=0.7, label="Magnitude Distribution", edgecolor="black")

    # Add vertical lines for RGB bin edges
    for i, edge in enumerate(rgb_bins[:-1]):
        plt.axvline(edge+0.05, color=['b', 'g', 'r'][i], linestyle='--', lw=2, label=f"Bin {i+1}")
    for i, edge in enumerate(rgb_bins[1:]):
        plt.axvline(edge-0.05, color=['b', 'g', 'r'][i], linestyle='--', lw=2)

    plt.xlabel("Magnitude", fontsize=14)
    plt.ylabel("Number of Stars", fontsize=14)
    plt.legend(fontsize=12)
    plt.title(f"Histogram of Magnitudes with {binning_method.capitalize()} RGB Bins", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # --------- Compute 2D Histogram for Sky Density ---------
    hist, xedges, yedges = np.histogram2d(ra, dec, bins=bins)

    # Determine contrast scaling
    non_zero = hist[hist > 0]
    vmin, vmax = np.percentile(non_zero, contrast)

    # --------- Plot Sky Density Map ---------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    norm_scale = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    im = ax.imshow(hist.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap=cmap_density, norm=norm_scale)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Star Density", fontsize=14)

    plt.xlabel("RA (deg)", fontsize=14)
    plt.ylabel("Dec (deg)", fontsize=14)
    plt.gca().invert_xaxis()
    plt.title("Sky Density of Selected Stars", fontsize=16)
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()

    # --------- Create False-Color RGB Composite ---------
    rgb_masks = [(mag >= rgb_bins[i]) & (mag < rgb_bins[i+1]) for i in range(3)]
    
    densities = []
    for mask in rgb_masks:
        ra_masked = ra[mask]
        dec_masked = dec[mask]
        hist, _, _ = np.histogram2d(ra_masked, dec_masked, bins=bins)
        densities.append(hist)

    # Normalize each channel
    def normalize_channel(channel):
        if channel.max() > 0:
            vmin, vmax = np.percentile(channel[channel > 0], [15, 98.5])
            return np.clip((channel - vmin) / (vmax - vmin), 0, 1)
        return channel

    r_channel = normalize_channel(densities[2].T)  # Faintest stars → Red
    g_channel = normalize_channel(densities[1].T)  # Medium brightness → Green
    b_channel = normalize_channel(densities[0].T)  # Brightest stars → Blue

    # Stack into RGB image
    rgb_image = np.dstack((r_channel, g_channel, b_channel))

    # --------- Plot False-Color RGB Composite ---------
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    im = ax.imshow(rgb_image, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    plt.xlabel("RA (deg)", fontsize=14)
    plt.ylabel("Dec (deg)", fontsize=14)
    plt.gca().invert_xaxis()
    plt.title(f"False-Color RGB Composite (Binning: {binning_method})", fontsize=16)
    plt.grid(alpha=0.3, linestyle="--")
    plt.show()


def plot_sky_density_healpy(gaia_data, nside=128, contrast=(5, 95), binning_method="linear", 
                             cmap_density="magma", cmap_rgb="plasma", log_scale=True):
    """
    Creates two visualizations using Healpy:
    1. An all-sky density map of stars (Galactic Coordinates).
    2. A false-color RGB composite image highlighting Galactic substructures.

    Parameters:
    -----------
    gaia_data : pd.DataFrame
        DataFrame containing 'ra', 'dec', and 'dered_G' magnitudes.
    nside : int, optional
        Healpy resolution parameter (default=128).
    contrast : tuple, optional
        Percentile range for contrast enhancement (default=(5, 95)).
    binning_method : str, optional
        Method for defining RGB bins ('linear' or 'normal'). Default is 'linear'.
    cmap_density : str, optional
        Colormap for the sky density map (default="magma").
    cmap_rgb : str, optional
        Colormap to test alternative RGB mappings (default="plasma").
    log_scale : bool, optional
        Whether to use a logarithmic color scale (default=True).

    Returns:
    --------
    None. Displays two all-sky maps.
    """

    ra = gaia_data['ra'].values
    dec = gaia_data['dec'].values
    mag = gaia_data['dered_G'].values

    # Convert RA/Dec to Galactic Coordinates
    theta = np.radians(90 - dec)  # Healpy uses theta = 90° - Dec
    phi = np.radians(ra)          # Healpy uses phi = RA

    # --------- Compute RGB Magnitude Bins ---------
    def compute_rgb_bins(mag, method="linear"):
        """
        Compute RGB bins using either 'linear' or 'normal' binning.

        Parameters:
            mag (array-like): Magnitude values.
            method (str): 'linear' for equal-width bins, 'normal' for normal distribution binning.

        Returns:
            list: Three bin edges for R/G/B assignment.
        """
        mag = np.array(mag)
        mag = mag[np.isfinite(mag)]  # Remove NaN values

        if method == "linear":
            return np.linspace(np.min(mag), np.max(mag), 4)  # Three bins

        elif method == "normal":
            mean, std = norm.fit(mag)
            return [np.percentile(mag, 16), np.percentile(mag, 50), np.percentile(mag, 84)]

        else:
            raise ValueError("Invalid method. Choose 'linear' or 'normal'.")

    # Compute RGB magnitude bins
    rgb_bins = compute_rgb_bins(mag, binning_method)

    # --------- Compute Healpy Binning for Sky Density ---------
    npix = hp.nside2npix(nside)
    density_map = np.zeros(npix)

    pix_indices = hp.ang2pix(nside, theta, phi)
    np.add.at(density_map, pix_indices, 1)  # Count stars per pixel

    # Determine contrast scaling
    non_zero = density_map[density_map > 0]
    vmin, vmax = np.percentile(non_zero, contrast)

    # --------- Plot All-Sky Density Map ---------
    fig = plt.figure(figsize=(10, 6), dpi=200)
    hp.mollview(density_map, norm=LogNorm(vmin=vmin, vmax=vmax) if log_scale else None,
                cmap=cmap_density, title="All-Sky Density of Selected Stars", unit="Star Density")
    hp.graticule(color="white", alpha=0.5)
    plt.show()

    # --------- Create False-Color RGB Composite ---------
    rgb_masks = [(mag >= rgb_bins[i]) & (mag < rgb_bins[i+1]) for i in range(3)]
    
    density_maps_rgb = []
    for mask in rgb_masks:
        density_map_channel = np.zeros(npix)
        pix_indices_channel = hp.ang2pix(nside, theta[mask], phi[mask])
        np.add.at(density_map_channel, pix_indices_channel, 1)
        density_maps_rgb.append(density_map_channel)

    # Normalize each channel
    def normalize_channel(channel):
        if channel.max() > 0:
            vmin, vmax = np.percentile(channel[channel > 0], [15, 98.5])
            return np.clip((channel - vmin) / (vmax - vmin), 0, 1)
        return channel

    r_channel = normalize_channel(density_maps_rgb[2])  # Faintest stars → Red
    g_channel = normalize_channel(density_maps_rgb[1])  # Medium brightness → Green
    b_channel = normalize_channel(density_maps_rgb[0])  # Brightest stars → Blue

    # Stack into RGB image
    rgb_image = np.dstack((r_channel, g_channel, b_channel))

    # --------- Plot False-Color RGB Composite ---------
    fig = plt.figure(figsize=(10, 6), dpi=200)
    hp.mollview(r_channel, cmap="Reds", title="Red Channel (Faintest Stars)")
    hp.mollview(g_channel, cmap="Greens", title="Green Channel (Intermediate Stars)")
    hp.mollview(b_channel, cmap="Blues", title="Blue Channel (Brightest Stars)")
    plt.show()

    # Final RGB composite
    fig = plt.figure(figsize=(10, 6), dpi=200)
    hp.mollview(r_channel + g_channel + b_channel, cmap=cmap_rgb,
                title=f"False-Color RGB Composite (Binning: {binning_method})")
    hp.graticule(color="white", alpha=0.5)
    plt.show()