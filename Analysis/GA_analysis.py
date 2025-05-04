import os
import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
from astropy.io import fits
import healpy as hp  

from galpy.potential import MWPotential2014  
from galpy.orbit import Orbit  

from dustmaps.sfd import SFDQuery  
from dustmaps.config import config

from astroquery.gaia import Gaia
from astroquery.sdss import SDSS
from astroquery.utils import tap

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tqdm.notebook import tqdm  
from tqdm import tqdm
from IPython.display import display


from astropy.io import votable
from astropy.table import Table
from astropy.io import fits





logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def query_gaia_halo_rgb(ra_min, ra_max, dec_min, dec_max, g_min = 14, g_max=23.5, parallax_max=1.0, ruwe_max=1.4, chunk_size=0.5, radial_velocity=True, save_path=None):
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
                       parallax, parallax_error, pmra, pmdec, radial_velocity, mh_gspphot, mh_gspspec,
                       teff_gspphot, logg_gspphot,
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


def reddening_correction(gaia_data_or_path, dustmaps_dir='dustmaps/'):
    """
    Applies Galactic extinction corrections to Gaia DR3 photometry using the 
    Schlegel, Finkbeiner & Davis (1998) (SFD) dust map and extinction coefficients 
    from Casagrande et al. (2021).
 
    The function adds new extinction-corrected columns to the dataset:
        - `dered_G`: Extinction-corrected G-band magnitude.
        - `dered_BP`: Extinction-corrected BP-band magnitude.
        - `dered_RP`: Extinction-corrected RP-band magnitude.
        - `dered_BP_RP`: Extinction-corrected BP-RP color index.
        - `M_G`: Absolute magnitude in the G-band, calculated using the extinction-corrected G-band magnitude
                 and the Bailer-Jones median photogeometric distance (`r_med_photogeo`).
 
    Parameters:
        gaia_data_or_path (str or pd.DataFrame): Either a DataFrame containing Gaia photometric 
                                                 and positional data or a file path to a FITS file.
        dustmaps_dir (str, optional): Directory where dustmaps are stored.

    Returns:
        None or pd.DataFrame: If a FITS file is provided, writes the corrected data to a new FITS file.
                              If a DataFrame is provided, returns the corrected DataFrame.
 
    Raises:
        ValueError: If required columns are missing from the input data.
    """

    # --------------- Set Dustmaps Data Directory ---------------
    config['data_dir'] = dustmaps_dir  
    sfd = SFDQuery() 

    # --------------- Handle FITS File Input ---------------
    if isinstance(gaia_data_or_path, str) and gaia_data_or_path.endswith(".fits"):
        input_fits = gaia_data_or_path
        # Define output file path
        output_fits = input_fits.replace(".fits", "_extinction_corrected.fits")

        # Load the data and convet to pandas df
        with fits.open(input_fits, memmap=True) as hdul:
            data = hdul[1].data 
            logging.info('Converting to a Pandas Dataframe...')
            # Allows for easy appending of new columns
            # This is a time consuming process - converting to a pandas df
            df = pd.DataFrame(data)


            # --------------- Check Required Columns Exist ---------------
            required_columns = {'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns in FITS file: {missing}")

            # --------------- Apply Extinction Correction ---------------
            logging.info('Applying extinction correction...')
            corrected_df = apply_extinction_correction(df, sfd)

            # --------------- Convert Back to FITS Format ---------------
            logging.info('Converting back to FITS format...')
            corrected_fits_data = corrected_df.to_records(index=False)

            # --------------- Save to New FITS File ---------------
            logging.info('Saving to new file...')
            primary_hdu = fits.PrimaryHDU()
            table_hdu = fits.BinTableHDU(data=corrected_fits_data)
            hdul_new = fits.HDUList([primary_hdu, table_hdu])
            hdul_new.writeto(output_fits, overwrite=True)

            logging.info(f"Extinction-corrected FITS file saved as: {output_fits}")
            return None

    # --------------- Handle Pandas DataFrame Input ---------------
    elif isinstance(gaia_data_or_path, pd.DataFrame):
        return apply_extinction_correction(gaia_data_or_path, sfd)

    else:
        raise ValueError("Input must be a pandas DataFrame or a FITS file path ending in .fits")


def apply_extinction_correction(batch_data, sfd):
    """
    Applies extinction correction directly to a data frame ontaining Gaia photometry.

    Parameters:
        batch_data (pd.DataFrame): DataFrame containing Gaia photometric and positional data.
        sfd (SFDQuery): Dust map query instance.

    Returns:
        np.ndarray: Updated NumPy structured array with extinction corrections applied.
            - `dered_G`: The extinction-corrected G-band magnitude.
            - `dered_BP`: The extinction-corrected BP-band magnitude.
            - `dered_RP`: The extinction-corrected RP-band magnitude.
            - `dered_BP_RP`: The extinction-corrected BP-RP color index.
            - `M_G`: The absolute magnitude in the Gaia G-band, computed from the extinction-corrected G-band magnitude and Bailer Jones median photogeometric distance (`r_med_photogeo`).
    """

    # Extract required columns as NumPy arrays
    ra = batch_data['ra']
    dec = batch_data['dec']
    g_mag = batch_data['phot_g_mean_mag']
    bp_mag = batch_data['phot_bp_mean_mag']
    rp_mag = batch_data['phot_rp_mean_mag']

    # Convert RA, Dec to SkyCoord for querying extinction maps
    coords = SkyCoord(ra=np.array(ra, dtype=float) * u.deg, 
                  dec=np.array(dec, dtype=float) * u.deg, 
                  frame="icrs")
    ebv = sfd(coords)  # Query dust map

    # Compute extinction coefficients (Casagrande et al. 2021)
    C = bp_mag - rp_mag
    R_G = 2.609 - 0.475 * C + 0.053 * C**2
    R_BP = 2.998 - 0.140 * C - 0.175 * C**2 + 0.062 * C**3
    R_RP = 1.689 - 0.059 * C

    # Apply extinction corrections
    batch_data['dered_G'] = g_mag - R_G * ebv
    batch_data['dered_BP'] = bp_mag - R_BP * ebv
    batch_data['dered_RP'] = rp_mag - R_RP * ebv
    batch_data['dered_BP_RP'] = batch_data['dered_BP'] - batch_data['dered_RP']

    # Compute Absolute Magnitude using extinction-corrected G-band
    batch_data["M_G"] = batch_data["dered_G"] - 5 * np.log10(batch_data["r_med_photogeo"]) + 5

    return batch_data


def rgb_filter(gaia_data_or_path, min_bp_rp=0.8, max_app_mag=18, max_abs_mag=5):
    """
    Filters Gaia data to select Red Giant Branch (RGB) stars based on 
    the BP-RP color index, apparent magnitude, and absolute magnitude.
    
    Loads the entire FITS file into memory, applies filtering, and writes the results to a new FITS file.

    Parameters:
        gaia_data_or_path (str or pd.DataFrame): Either a DataFrame containing Gaia photometric data 
                                                 or a file path to a FITS file.
        min_bp_rp (float, optional): Minimum BP-RP color index (default=0.8).
        max_app_mag (float, optional): Maximum apparent magnitude in dereddened G-band (default=18).
        max_abs_mag (float, optional): Maximum absolute magnitude in dereddened G-band (default=5).

    Returns:
        pd.DataFrame or None: 
            - If a DataFrame is provided, returns the filtered DataFrame.
            - If a FITS file is provided, writes the filtered data to a new FITS file.
    """

    # --------------- Handle FITS File Input ---------------
    if isinstance(gaia_data_or_path, str) and gaia_data_or_path.endswith(".fits"):
        input_fits = gaia_data_or_path
        output_fits = input_fits.replace(".fits", "_filtered.fits")

        # Load full FITS table into memory
        with fits.open(input_fits, memmap=True) as hdul:

            data = hdul[1].data
            total_stars = hdul[1].header['NAXIS2']
            logging.info(f'Loaded {total_stars} from FITS File ...')
            logging.info('Converting to a Pandas Dataframe...')
            df = pd.DataFrame(data)
            # Ensure all numerical columns are little-endian
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(df[col].dtype.newbyteorder("="))

        # Ensure columns required for analysis exist
        required_columns = {'dered_BP_RP', 'dered_G', 'M_G'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in FITS file: {missing_columns}")

        # Apply filtering and count statistics
        logging.info("Applying RGB filter...")
        filtered_df, counts = _apply_rgb_filter(df, min_bp_rp, max_app_mag, max_abs_mag)

        total_bp_rp = counts["bp_rp"]
        total_abs_mag = counts["abs_mag"]
        total_app_mag = counts["app_mag"]
        total_b_cut = counts["b_cut"]
        total_passed = counts["passed"]

        # Print final summary
        logging.info(
            f"\nTotal stars before filtering: {total_stars}\n"
            f"Stars passing Galactic latitude cut (|b| > 10°): {total_b_cut} ({total_b_cut/total_stars:.2%})"
            f"Stars passing BP-RP color filter: {total_bp_rp} ({total_bp_rp/total_stars:.2%})\n"
            f"Stars passing apparent magnitude filter: {total_app_mag} ({total_app_mag/total_stars:.2%})\n"
            f"Stars passing absolute magnitude filter: {total_abs_mag} ({total_abs_mag/total_stars:.2%})\n"
            f"Stars passing all filters: {total_passed} ({total_passed/total_stars:.2%})\n"
            f"Halo RGB filtered FITS file saved as: {output_fits}"
        )

        # Convert to FITS format and save
        logging.info("Saving filtered data to new FITS file...")
        corrected_fits_data = filtered_df.to_records(index=False)
        primary_hdu = fits.PrimaryHDU()
        table_hdu = fits.BinTableHDU(data=corrected_fits_data)
        hdul_new = fits.HDUList([primary_hdu, table_hdu])
        hdul_new.writeto(output_fits, overwrite=True)

        logging.info(f"Halo RGB filtered FITS file saved as: {output_fits}")
        return None  

    # --------------- Handle Pandas DataFrame Input ---------------
    elif isinstance(gaia_data_or_path, pd.DataFrame):
        filtered_df, _ = _apply_rgb_filter(gaia_data_or_path, min_bp_rp, max_app_mag, max_abs_mag)
        return filtered_df

    else:
        raise ValueError("Input must be a pandas DataFrame or a FITS file path ending in .fits")


def _apply_rgb_filter(batch_data, min_bp_rp, max_app_mag, max_abs_mag):
    """
    Applies RGB filtering to a batch of Gaia data using NumPy for speed.

    Parameters:
        batch_data (pd.DataFrame): DataFrame containing Gaia photometric data.
        min_bp_rp (float): Minimum BP-RP color index.
        max_app_mag (float): Maximum apparent magnitude.
        max_abs_mag (float): Maximum absolute magnitude.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only stars meeting the criteria.
        dict: Dictionary with counts for each filtering step.
    """

    bp_rp = batch_data["dered_BP_RP"]
    dered_G = batch_data["dered_G"]
    abs_mag = batch_data["M_G"]
    b_lat = batch_data["b"]  # Galactic latitude in degrees

    # Apply individual filters
    mask_bp_rp = (bp_rp > min_bp_rp)
    mask_app_mag = dered_G < max_app_mag
    mask_abs_mag = abs_mag < max_abs_mag
    mask_b_cut = np.abs(b_lat) > 10  # Filter for |b| > 10 degrees

    # Combine all filters
    mask_total = mask_bp_rp & mask_app_mag & mask_abs_mag & mask_b_cut

    # Count statistics
    counts = {
        "bp_rp": np.sum(mask_bp_rp),
        "app_mag": np.sum(mask_app_mag),
        "abs_mag": np.sum(mask_abs_mag),
        "b_cut": np.sum(mask_b_cut),
        "passed": np.sum(mask_total),
    }

    # Return only rows that pass all filters
    return batch_data[mask_total], counts


def add_galpy_orbital_parameters(gaia_data_or_path):
    """
    Computes and adds orbital parameters using galpy for Gaia stars using SkyCoord.
    Accepts either a FITS file or a Pandas DataFrame.

    Parameters:
        gaia_data_or_path (str or pd.DataFrame): Either a DataFrame containing Gaia data or a file path to a FITS file.

    Returns:
        None or pd.DataFrame:
            - If a FITS file is provided, saves the updated data to a new FITS file with `_galpy.fits` suffix.
            - If a DataFrame is provided, returns the modified DataFrame.
    
    Raises:
        ValueError: If required columns are missing from the input data.
    """

    if isinstance(gaia_data_or_path, str) and gaia_data_or_path.endswith(".fits"):
        input_fits = gaia_data_or_path
        output_fits = input_fits.replace(".fits", "_galpy.fits")

        with fits.open(input_fits, memmap=True) as hdul:
            data = hdul[1].data
            df = pd.DataFrame(data)

        # Ensure required columns exist
        required_columns = {'ra', 'dec', 'r_med_photogeo', 'pmra', 'pmdec', 'radial_velocity'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns in FITS file: {missing}")

        # Compute orbital parameters
        df = _compute_galpy_orbital_parameters(df)

        # Convert back to FITS and save
        corrected_fits_data = df.to_records(index=False)
        primary_hdu = fits.PrimaryHDU()
        table_hdu = fits.BinTableHDU(data=corrected_fits_data)
        hdul_new = fits.HDUList([primary_hdu, table_hdu])
        hdul_new.writeto(output_fits, overwrite=True)

        print(f"Galpy orbital parameters added. Updated FITS file saved as: {output_fits}")
        return None

    elif isinstance(gaia_data_or_path, pd.DataFrame):
        # Ensure required columns exist
        required_columns = {'ra', 'dec', 'r_med_photogeo', 'pmra', 'pmdec', 'radial_velocity'}
        if not required_columns.issubset(gaia_data_or_path.columns):
            missing = required_columns - set(gaia_data_or_path.columns)
            raise ValueError(f"Missing required columns in DataFrame: {missing}")

        return _compute_galpy_orbital_parameters(gaia_data_or_path)

    else:
        raise ValueError("Input must be a pandas DataFrame or a FITS file path ending in .fits")


def _compute_galpy_orbital_parameters(df):
    """
    Computes Galactic orbital parameters using galpy and adds them to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing Gaia kinematic data.

    Returns:
        pd.DataFrame: Updated DataFrame with new orbital parameter columns.
    """

    df["energy"] = np.nan
    df["Lz"] = np.nan
    df["R_gal"] = np.nan

    for i in tqdm(range(len(df)), desc="Processing stars", unit="star"):
        try:
            # Convert Gaia observables into Astropy SkyCoord
            c = SkyCoord(
                ra=df.loc[i, "ra"] * u.deg,
                dec=df.loc[i, "dec"] * u.deg,
                distance=df.loc[i, "r_med_photogeo"] * u.pc, 
                pm_ra_cosdec=df.loc[i, "pmra"] * u.mas / u.yr,
                pm_dec=df.loc[i, "pmdec"] * u.mas / u.yr,
                radial_velocity=df.loc[i, "radial_velocity"] * u.km / u.s
            )

            # Convert to Galpy Orbit object
            orbit = Orbit(c)
            # Compute and store the key orbital parameters
            df.at[i, "energy"] = orbit.E(pot=MWPotential2014)
            df.at[i, "Lz"] = orbit.Lz()
            df.at[i, "R_gal"] = orbit.R()

        except Exception as e:
            print(f"Failed to compute orbit for row {i}: {e}")

    return df


def apogee_metallicities(gaia_cluster_json, apogee_fits_path = 'data/ApogeeDR17_allStarLite.fits', harris_data_path = 'data/harris2010_mwgc.dat', matched_star_threshold=20):
    """
    Processes Gaia cluster data by cross-matching with APOGEE DR17 metallicity data 
    and integrating Harris 2010 globular cluster metallicity values.

    This function loads Gaia-based stellar clusters, extracts associated Gaia DR3 
    source IDs, and attempts to match them with stars from the APOGEE DR17 catalog 
    based on their source IDs. It then calculates statistical metallicity properties 
    for each matched cluster and integrates [Fe/H] values from the Harris 2010 
    globular cluster catalog where available.

    Parameters:
    -----------
    gaia_cluster_json : str
        Path to a JSON file containing matched clusters and their corresponding 
        Gaia EDR3 source IDs.
    apogee_fits_path : str, optional
        Path to the APOGEE DR17 allStarLite FITS file (default: 'data/ApogeeDR17_allStarLite.fits').
    harris_data_path : str, optional
        Path to the Harris 2010 metallicity catalog for Milky Way globular clusters 
        (default: 'data/harris2010_mwgc.dat').
    matched_star_threshold : int, optional
        Minimum number of matched stars required for a cluster to be included in 
        the final results (default: 20).

    Returns:
    --------
    filtered_df : pd.DataFrame
        A DataFrame containing matched clusters along with their APOGEE and Harris 
        metallicity values. The table includes:
        - Cluster Name
        - Number of matched stars
        - Mean and standard deviation of [Fe/H] (iron abundance)
        - Mean and standard deviation of [M/H] (total metallicity)
        - Mean and standard deviation of [α/M] (alpha-element enhancement)
        - Harris 2010 metallicity ([Fe/H]) if available

    The function also prints:
        - A full summary of all clusters matched with APOGEE metallicities.
        - A filtered table including only clusters that meet the `matched_star_threshold` 
          and have Harris 2010 metallicity values."
    """

    # Load Gaia Clusters from Notebook 6's - These give the cluster limits in l, b, pmra and pmdec and the Gaia Sources that belong to this from the Gaia EDR3 data
    # These Gaia IDs are from the resepective data sets used and not from all stars in the GAIA EDR3 data
    with open(gaia_cluster_json, "r") as f:
        matched_clusters = json.load(f)

    # Extract Gaia source IDs from matched clusters
    cluster_stellar_ids = {cluster_name: details["gaia_source_ids"] for cluster_name, details in matched_clusters.items()}

    all_gaia_ids = [star_id for ids in cluster_stellar_ids.values() for star_id in ids]

    logging.info(f"Total Gaia IDs that will be attempted to match with APOGEE: {len(all_gaia_ids)}")

    # Load the Apogee Data from file - all star lite catalogue
    with fits.open(apogee_fits_path, memmap=True) as hdul:
        apogee_data = hdul[1].data 

    # Extract data and convert to DataFrame - Convert from <f8 and <i8 to float32 and int64 respectively as big - endian not supported
    apogee_df = pd.DataFrame({
        "gaia_source_id": apogee_data["GAIAEDR3_SOURCE_ID"].byteswap().view(np.int64),
        "fe_h": apogee_data["FE_H"].byteswap().view(np.float32),
        "m_h": apogee_data["M_H"].byteswap().view(np.float32),
        "alpha_m": apogee_data["ALPHA_M"].byteswap().view(np.float32)
    })

    # Remove any rows with missing Gaia source IDs 
    apogee_df.dropna(subset=["gaia_source_id"], inplace=True)

    logging.info(f"Successfully loaded {len(apogee_df)} stars from APOGEE DR17.")

    # Mapp the apoogee data to the gaia star IDs from the clusters
    gaia_stars_df = pd.DataFrame({"gaia_source_id": all_gaia_ids})
    matched_metallicities = gaia_stars_df.merge(apogee_df, on="gaia_source_id", how="left")

    logging.info(f"Matched {len(matched_metallicities.dropna())} Gaia stars with APOGEE metallicities: While trying to find {len(all_gaia_ids)} Gaia IDS in {len(apogee_df)} APOGEE items.")

    # Compute Cluster Statistics
    cluster_stats = []

    logging.info(f"Calculating statistics for each cluster with available apogee data.")

    # For each cluster, find the matched stars and compute statistics for Fe_H, M_H, and Alpha_M
    for cluster_name, details in matched_clusters.items():
        # Find stars that are matched with APOGEE metalicites
        matched_star_ids = set(details["gaia_source_ids"]) & set(matched_metallicities.dropna()["gaia_source_id"])
        cluster_data = matched_metallicities[matched_metallicities["gaia_source_id"].isin(matched_star_ids)]


        # Compute statistics - Count, Mean, and Std Dev
        matched_count = len(cluster_data)
        fe_h_mean, fe_h_std = cluster_data["fe_h"].mean(), cluster_data["fe_h"].std()
        m_h_mean, m_h_std = cluster_data["m_h"].mean(), cluster_data["m_h"].std()
        alpha_m_mean, alpha_m_std = cluster_data["alpha_m"].mean(), cluster_data["alpha_m"].std()
        cluster_stats.append([cluster_name, matched_count, fe_h_mean, fe_h_std, m_h_mean, m_h_std, alpha_m_mean, alpha_m_std])

    # Convert to DataFrame
    cluster_stats_df = pd.DataFrame(cluster_stats, columns=[
        "Cluster Name", "Matched Stars", 
        "Mean Fe_H", "Std Dev Fe_H",
        "Mean M_H", "Std Dev M_H",
        "Mean Alpha_M", "Std Dev Alpha_M"
    ])

    ### Load the Harris 2010 Metallicities from file
    df_table1 = pd.read_fwf(harris_data_path, skiprows=70, nrows=158)
    df_table2 = pd.read_fwf(harris_data_path, skiprows=250, nrows=158)
    glob_clust_harris = pd.merge(df_table1, df_table2, on="ID", suffixes=("_table1", "_table2"))

    ### Add Harris Metallicities to table if available
    glob_clust_harris["ID"] = glob_clust_harris["ID"].astype(str)
    cluster_stats_df = cluster_stats_df.merge(
        glob_clust_harris[["ID", "[Fe/H]"]], 
        left_on="Cluster Name", 
        right_on="ID", 
        how="left"
    )

    logging.info(f"Successfully merged Harris 2010 metallicities with cluster statistics for those stars available.")
    cluster_stats_df.rename(columns={"[Fe/H]": "Harris_Fe_H"}, inplace=True)
    cluster_stats_df.drop(columns=["ID"], inplace=True)

    # Display a filtered DataFrame where Matched Stars > value and Harris Fe_H is available
    filtered_df = cluster_stats_df[
        (cluster_stats_df["Matched Stars"] > matched_star_threshold) & cluster_stats_df["Harris_Fe_H"].notna()
    ]

    # Display Results
    print("\nFull Cluster Statistics Table:")
    display(cluster_stats_df)

    print(f"\nFiltered Clusters (Matched Stars > {matched_star_threshold} and valid Harris_Fe_H):")
    display(filtered_df)

    return cluster_stats_df

### RGB Stars filtering

def plot_sky_density_healpy(gaia_data, nside=128, contrast=(5, 95), vmin = 20 , vmax = 40, binning_method="linear", 
                             cmap_density="magma", cmap_rgb="plasma", log_scale=True):
    """
    Generates multiple sky density visualizations using HEALPix:
    
    1. **All-sky density map** (Mollweide projection) - visualizing star distribution in Galactic coordinates.
    2. **False-color RGB composite** (Aitoff projection) - highlighting stellar populations in different brightness bins.
    3. **Histogram of magnitudes** - showing how stars are divided into RGB bins.
    4. **Histogram of sky density** - illustrating the distribution of star densities across HEALPix pixels.
    5. **Rectangular projection of the RGB composite** - displaying Galactic coordinates without distortion.

    Parameters:
    -----------
    gaia_data : pd.DataFrame
        Pandas DataFrame containing:
        - 'l' : Galactic longitude (degrees)
        - 'b' : Galactic latitude (degrees)
        - 'dered_G' : Dereddened Gaia G-band magnitude
    nside : int, optional (default=128)
        HEALPix resolution parameter. Higher values provide finer detail.
    contrast : tuple, optional (default=(5, 95))
        Percentile range for adjusting color contrast.
    vmin, vmax : float, optional (default=20, 40)
        Minimum and maximum density values for histogram scaling.
    binning_method : str, optional (default="linear")
        Binning method for RGB classification. Options:
        - "linear" : Equal-width bins.
        - "normal" : Bins based on Gaussian distribution of magnitudes.
        - tuple : Custom bin edges.
    cmap_density : str, optional (default="magma")
        Colormap for the density map.
    cmap_rgb : str, optional (default="plasma")
        Alternative colormap for RGB mapping.
    log_scale : bool, optional (default=True)
        If True, applies logarithmic scaling to enhance density contrast.

    Returns:
    --------
    None. Generates and displays multiple sky density plots.
    
    Notes:
    ------
    - Uses HEALPix for efficient pixelization of celestial sphere data.
    - RGB composite maps classify stars into three magnitude bins.
    - The rectangular projection enables direct interpretation of Galactic coordinates.
    """

    l = gaia_data['l'].values  # Galactic longitude
    b = gaia_data['b'].values  # Galactic latitude
    mag = gaia_data['dered_G'].values  # Magnitude

    # Convert RA/Dec to Galactic Coordinates
    theta = np.radians(90 - b)  # Healpy uses theta = 90° - Dec
    phi = np.radians(l)          # Healpy uses phi = RA

    mag_low, mag_high = np.percentile(mag, 1), np.percentile(mag, 99)

    # --------- Compute Healpy Binning for Sky Density ---------
    npix = hp.nside2npix(nside)
    density_map = np.zeros(npix)

    pix_indices = hp.ang2pix(nside, theta, phi)
    np.add.at(density_map, pix_indices, 1)  # Count stars per pixel

    # Determine contrast scaling
    non_zero = density_map[density_map > 0]
    # vmin, vmax = np.percentile(non_zero, contrast[0]), np.percentile(non_zero, contrast[1])

    # --------- Compute RGB Magnitude Bins ---------
    def compute_rgb_bins(mag, method= binning_method):
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
            middle_bins = np.linspace(mag_low, mag_high, 4)  # Three bins
            end_bins = np.linspace(np.min(mag), np.max(mag), 4)  
            middle_bins[0] = end_bins[0]
            middle_bins[3] = end_bins[3]

            return middle_bins  # Three bins

        elif method == "normal":
            return [np.min(mag), np.percentile(mag, 33), np.percentile(mag, 66), np.max(mag)]
        
        elif isinstance(method, tuple):
            return [np.min(mag), method[0], method[1], np.max(mag)]

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


    # --------- Plot Histogram of Density Map ---------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    plt.hist(density_map, bins=40, color='gray', alpha=0.7, label="Density Distribution", edgecolor="black")
    plt.yscale('log')

    # Add vertical lines for RGB bin edges
    plt.axvline(vmin, color='r', linestyle='--', lw=2, label=f"Saturation Boundary")
    plt.axvline(vmax, color='r', linestyle='--', lw=2)


    plt.xlabel("Density of Bins", fontsize=14)
    plt.ylabel("Number of Stars", fontsize=14)
    plt.legend(fontsize=12)
    plt.title(f"Histogram of Density Map", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


    # --------- Plot All-Sky Density Map ---------
    fig = plt.figure(figsize=(10, 6), dpi=200)
    clipped_density_map = np.where(density_map < vmin, vmin, density_map)
    clipped_density_map = np.where(clipped_density_map > vmax, vmax, clipped_density_map)
    if log_scale: 
        clipped_density_map = np.log(clipped_density_map +1)
        hp.mollview(clipped_density_map,
                    cmap=cmap_density, title="All-Sky Density of Selected Stars", unit="Log Star Density")
    else: 
        hp.mollview(clipped_density_map,
                    cmap=cmap_density, title="All-Sky Density of Selected Stars", unit="Star Density")
    
    hp.graticule(color="white", alpha=0.5)
    plt.show()


    fig = plt.figure(figsize=(10, 6), dpi=200)

    # --------- Create False-Color RGB Composite : Globe View ---------
    rgb_masks = [(mag >= rgb_bins[i]) & (mag < rgb_bins[i+1]) for i in range(3)]
    
    density_maps_rgb = []
    for mask in rgb_masks:
        density_map_channel = np.zeros(npix)
        pix_indices_channel = hp.ang2pix(nside, theta[mask], phi[mask])
        np.add.at(density_map_channel, pix_indices_channel, 1)
        density_maps_rgb.append(density_map_channel)

    # Normalize each channel
    def normalize_channel(channel):
        """ Normalize a HEALPix map to the [0,1] range. """
        if channel.max() > 0:
            vmin, vmax = np.percentile(channel[channel > 0], [contrast[0], contrast[1]])
            return np.clip((channel - vmin) / (vmax - vmin), 0, 1)
        return channel

    # Normalize each RGB channel separately
    r_channel = normalize_channel(density_maps_rgb[2])  # Faintest stars (Red)
    g_channel = normalize_channel(density_maps_rgb[1])  # Medium brightness (Green)
    b_channel = normalize_channel(density_maps_rgb[0])  # Brightest stars (Blue)

    # Stack into RGB image (N_pixels, 3)
    rgb_healpix_map = np.stack([r_channel, g_channel, b_channel], axis=-1)  # Shape (N_pixels, 3)

    # Convert HEALPix pixel indices to sky coordinates
    nside = hp.get_nside(r_channel)
    theta, phi = hp.pix2ang(nside, np.arange(rgb_healpix_map.shape[0]))

    # Convert to RA/Dec (or Galactic coordinates)
    # Convert HEALPix angles to Galactic Coordinates
    l = np.degrees(phi)  # Galactic longitude
    b = 90 - np.degrees(theta)  # Galactic latitude

    # Ensures -180° to +180° range
    l = -((l + 180) % 360 - 180)  

    # --------- Create False-Color RGB Composite : Globe View ---------

    # Convert HEALPix map to an Aitoff projection
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(111, projection="aitoff")

    # Scatter plot to display the RGB colors
    sc = ax.scatter(
        np.radians(l), 
        np.radians(b),
        c=rgb_healpix_map,
        marker="o",
        s=5,  
        edgecolor="none"
    )

    ax.set_xlabel("Galactic Longitude", fontsize=14)
    ax.set_ylabel("Galactic Latitude", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # --------- Create False-Color RGB Composite : Rectangular View ---------

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    # Scatter plot of RGB composite with correct orientation
    sc = ax.scatter(
        l, b, 
        c=rgb_healpix_map,
        marker="o",
        s=200,
        edgecolor="none") 

    # Set axis labels and title
    ax.set_xlabel("Galactic Longitude (l)", fontsize=14)
    ax.set_ylabel("Galactic Latitude (b)", fontsize=14)

    # Set Galactic coordinate limits
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.show()



def plot_sky_density_FS(gaia_data, bins=200, contrast = (0,100), binning_method="linear", 
                     cmap_density="magma", density_bins = 7 , log_scale=True):
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
            return [np.min(mag), np.percentile(mag, 33), np.percentile(mag, 66), np.max(mag)]
        
        elif isinstance(method, tuple):
            return [np.min(mag), method[0], method[1], np.max(mag)]

        else:
            raise ValueError("Invalid method. Choose 'linear' or 'normal' or custom - tuple.")
        
    
    # Compute RGB magnitude bins
    rgb_bins = compute_rgb_bins(mag, binning_method)


    # --------- Compute 2D Histogram for Sky Density ---------
    hist, xedges, yedges = np.histogram2d(ra, dec, bins=bins)
    # Determine contrast scaling
    non_zero = hist[hist > 0]
    vmin, vmax = np.percentile(non_zero, contrast[0]), np.percentile(non_zero, contrast[1])

    print(f"Contrast Limits: {vmin:.2f} to {vmax:.2f}")

    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # --------- Left: Histogram of Magnitude with RGB Bins ---------
    axes[0].hist(mag, bins=40, color='gray', alpha=0.7, label="Magnitude Distribution", edgecolor="black")

    # Add vertical lines for RGB bin edges
    for i, edge in enumerate(rgb_bins[:-1]):
        axes[0].axvline(edge + 0.05, color=['b', 'g', 'r'][i], linestyle='--', lw=2, label=f"Bin {i+1}")
    for i, edge in enumerate(rgb_bins[1:]):
        axes[0].axvline(edge - 0.05, color=['b', 'g', 'r'][i], linestyle='--', lw=2)

    axes[0].set_xlabel("Magnitude", fontsize=14)
    axes[0].set_ylabel("Number of Stars", fontsize=14)
    axes[0].legend(fontsize=12)
    if isinstance(binning_method, tuple):
        axes[0].set_title("Histogram of Magnitudes with Custom RGB Bins", fontsize=16)
    else:
        axes[0].set_title(f"Histogram of Magnitudes with {binning_method.capitalize()} RGB Bins", fontsize=16)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # --------- Right: Histogram of Density Map ---------
    axes[1].hist(hist.ravel(), bins=density_bins, color='gray', alpha=0.7, label="Density Distribution", edgecolor="black", rwidth=1)
    axes[1].set_yscale('log')

    # Add vertical lines for contrast limits
    axes[1].axvline(vmin, color='r', linestyle='--', lw=2, label="Saturation Boundary")
    axes[1].axvline(vmax, color='r', linestyle='--', lw=2)

    axes[1].set_xlabel("Density of Bins", fontsize=14)
    axes[1].set_ylabel("Number of Stars", fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].set_title("Histogram of Density Map", fontsize=16)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()





    # Create figure with 1 row and 2 columns for better side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # --------- Left: Sky Density Map ---------
    norm_scale = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    cut_hist = np.clip(hist, vmin, vmax)  # Clip histogram values for better contrast

    im1 = axes[0].imshow(cut_hist.T, origin="lower", 
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        cmap=cmap_density, norm=norm_scale)

    # Labels and Title Formatting
    axes[0].set_xlabel("Right Ascension (°)", fontsize=18)
    axes[0].set_ylabel("Declination (°)", fontsize=18)
    axes[0].invert_xaxis()
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].grid(alpha=0.3, linestyle="--")
    


    # --------- Right: False-Color RGB Composite ---------
    rgb_masks = [(mag >= rgb_bins[i]) & (mag < rgb_bins[i+1]) for i in range(3)]
    densities = []

    for mask in rgb_masks:
        ra_masked = ra[mask]
        dec_masked = dec[mask]
        hist, _, _ = np.histogram2d(ra_masked, dec_masked, bins=bins)
        densities.append(hist)

    # Normalize each channel
    def normalize_channel(channel):
        """Normalize an RGB channel while preserving structure."""
        if channel.max() > 0:
            return np.clip((channel - vmin/2) / (vmax/2 - vmin/2), 0, 1)
        return channel

    # Apply normalization to RGB channels
    r_channel = normalize_channel(densities[2].T)  # Faintest stars (Red)
    g_channel = normalize_channel(densities[1].T)  # Medium brightness (Green)
    b_channel = normalize_channel(densities[0].T)  # Brightest stars (Blue)

    # Stack into an RGB image
    rgb_image = np.dstack((r_channel, g_channel, b_channel))

    im2 = axes[1].imshow(rgb_image, origin="lower", 
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # Labels and Title Formatting
    axes[1].set_xlabel("Right Ascension (°)", fontsize=18)
    axes[1].set_ylabel("Declination (°)", fontsize=18)
    axes[1].invert_xaxis()
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    title_text = "False-Color RGB Composite"


    # Adjust layout for publication quality
    plt.tight_layout()
    plt.show()


    ### RGB Stars filtering

def plot_sky_density_proper_motion(gaia_data, pm_cuts=[1,3.5], cmap="inferno", min_count = 7, min_count_color = 5, max_count_color = 95):
    """
    Generates multiple sky density visualizations using HEALPix:
    
    1. **All-sky density map** (Mollweide projection) - visualizing star distribution in Galactic coordinates.
    2. **False-color RGB composite** (Aitoff projection) - highlighting stellar populations in different brightness bins.
    3. **Histogram of magnitudes** - showing how stars are divided into RGB bins.
    4. **Histogram of sky density** - illustrating the distribution of star densities across HEALPix pixels.
    5. **Rectangular projection of the RGB composite** - displaying Galactic coordinates without distortion.

    Parameters:
    -----------
    gaia_data : pd.DataFrame
        Pandas DataFrame containing:
        - 'l' : Galactic longitude (degrees)
        - 'b' : Galactic latitude (degrees)
        - 'dered_G' : Dereddened Gaia G-band magnitude
    nside : int, optional (default=128)
        HEALPix resolution parameter. Higher values provide finer detail.
    contrast : tuple, optional (default=(5, 95))
        Percentile range for adjusting color contrast.
    vmin, vmax : float, optional (default=20, 40)
        Minimum and maximum density values for histogram scaling.
    binning_method : str, optional (default="linear")
        Binning method for RGB classification. Options:
        - "linear" : Equal-width bins.
        - "normal" : Bins based on Gaussian distribution of magnitudes.
        - tuple : Custom bin edges.
    cmap_density : str, optional (default="magma")
        Colormap for the density map.
    cmap_rgb : str, optional (default="plasma")
        Alternative colormap for RGB mapping.
    log_scale : bool, optional (default=True)
        If True, applies logarithmic scaling to enhance density contrast.

    Returns:
    --------
    None. Generates and displays multiple sky density plots.
    
    Notes:
    ------
    - Uses HEALPix for efficient pixelization of celestial sphere data.
    - RGB composite maps classify stars into three magnitude bins.
    - The rectangular projection enables direct interpretation of Galactic coordinates.
    """

    # ------------------------------
    # GLOBAL PLOTTING PARAMETERS
    # ------------------------------
    figsize_wide = (9, 6)
    figsize_moll = (11, 6)
    fontsize_labels = 16
    fontsize_ticks = 14
    fontsize_title = 18
    tick_length = 6
    tick_width = 1.5

    # ------------------------------
    # DATA EXTRACTION
    # ------------------------------
    l = gaia_data['l'].values
    b = gaia_data['b'].values
    ra = gaia_data['ra'].values
    dec = gaia_data['dec'].values
    mag = gaia_data['dered_G'].values
    pm = gaia_data['pm'].values
    dered_G = gaia_data['dered_G'].values
    dered_BP_RP = gaia_data['dered_BP_RP'].values

    ### --- Plot 1 ----
    ra_range=[0,360]
    dec_range=np.array([-90,90])
    rev_rar_range=np.flip(ra_range)
    nra = int(1*190)
    ndec = int(1*95)

    fig, ax = plt.subplots(figsize=figsize_wide)

    den, xedges, yedges = np.histogram2d(ra, dec, bins=(nra, ndec))
    # For bins that are empty set to a very small value
    w0 = den == 0
    wn0 = den !=0
    den[w0] = 1e-6
    # Apply log scaling
    den_log = np.log10(den)

    # Remove low-count noise
    den_log[den < min_count] = np.nan  # cleaner background


    ax.pcolormesh(xedges, yedges, den_log.T, cmap= cmap , vmin = np.nanpercentile(den_log[wn0], 10), \
                vmax = np.nanpercentile(den_log[wn0], 90))

    ax.set_xlim(rev_rar_range)
    ax.set_ylim(dec_range)
    ax.set_xlabel('Right Ascension (RA) [deg]', fontsize=fontsize_labels)
    ax.set_ylabel('Declination (Dec) [deg]', fontsize=fontsize_labels)
    ax.set_title('All-Sky Density in Equatorial Coordinates', fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()



    ax.pcolormesh(xedges, yedges, den_log.T, cmap= cmap , vmin = np.nanpercentile(den_log[wn0], 10), \
                vmax = np.nanpercentile(den_log[wn0], 90))

    ax.set_xlim(rev_rar_range)
    ax.set_ylim(dec_range)
    ax.set_xlabel('Right Ascension (RA) [deg]', fontsize=fontsize_labels)
    ax.set_ylabel('Declination (Dec) [deg]', fontsize=fontsize_labels)
    ax.set_title('All-Sky Density in Equatorial Coordinates', fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()


    ###  --- Plot 2 ----
    ra_range_shifted = [-180, 180]  # updated range

    # Shift longitudes so Galactic center (l=0) is centered
    l_shifted = (-l + 180) % 360 - 180
    b_vals = b

    # Plot
    fig, ax = plt.subplots(figsize=figsize_wide)

    den, xedges, yedges = np.histogram2d(l_shifted, b_vals, bins=(nra, ndec), range=[ra_range_shifted, dec_range])
    

    ax.pcolormesh(xedges, yedges, den.T, cmap=cmap, vmin = np.nanpercentile(den[wn0], 5), \
                vmax = np.nanpercentile(den[wn0], 90))

    # Axes

    ax.set_xlim(ra_range_shifted)
    ax.set_ylim(dec_range)
    ax.set_xlabel("Galactic Longitude (l) [deg]", fontsize=fontsize_labels)
    ax.set_ylabel("Galactic Latitude (b) [deg]", fontsize=fontsize_labels)
    ax.set_title("Galactic Density Map (l = 0° Centered)", fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()



    ### --- Plot 1 AGAIN ----
    ra_range=[0,360]
    dec_range=np.array([-90,90])
    rev_rar_range=np.flip(ra_range)
    nra = int(1*360)
    ndec = int(1*180)

    fig, ax = plt.subplots(figsize=figsize_wide)

    den, xedges, yedges = np.histogram2d(ra, dec, bins=(nra, ndec))
    # For bins that are empty set to a very small value
    w0 = den == 0
    wn0 = den !=0
    den[w0] = 1e-6
    # Apply log scaling
    den_log = np.log10(den)


    ax.pcolormesh(xedges, yedges, den_log.T, cmap= cmap , vmin = np.nanpercentile(den_log[wn0], 10), \
                vmax = np.nanpercentile(den_log[wn0], 90))

    ax.set_xlim(rev_rar_range)
    ax.set_ylim(dec_range)
    ax.set_xlabel('Right Ascension (RA) [deg]', fontsize=fontsize_labels)
    ax.set_ylabel('Declination (Dec) [deg]', fontsize=fontsize_labels)
    ax.set_title('All-Sky Density in Equatorial Coordinates', fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()
    


    ### ---- Plot 3 ----
    # Create three sample grayscale images (arrays)
    # Bins set up
    pm_bins =  np.linspace(pm_cuts[0], pm_cuts[1], 4)  # Generate three bin edges

    images = []
    for i in range(3):
        # Set the bin edges
        filter_pm = (pm>pm_bins[i]) & (pm<pm_bins[i+1]) 
        # Bin
        den, x, y = np.histogram2d(np.array(ra[filter_pm]), np.array(dec[filter_pm]), bins=(nra, ndec))
        wn0 = den !=0
        log_den = np.log10(den)

        # Apply log scaling
        p5, p95 = np.percentile(log_den[wn0], [min_count_color, max_count_color])
        img_cur = np.clip(log_den, p5, p95)
        img_cur = (img_cur - p5) / (p95 - p5)
        # compute the image
        images.append(img_cur.T)

    # Stack the grayscale images to create an RGB image
    rgb_image = np.dstack((images[0], images[1], images[2]))

    # Plot
    rgb_image_flipped = np.fliplr(rgb_image)
    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.imshow(rgb_image, origin='lower', extent=[rev_rar_range[0], rev_rar_range[1], dec_range[0], dec_range[1]], aspect='auto', interpolation='bilinear')

    ax.set_xlim(rev_rar_range)
    ax.set_ylim(dec_range)

    ax.set_xlabel('RA [deg]', fontsize=fontsize_labels)
    ax.set_ylabel('Dec [deg]', fontsize=fontsize_labels)
    ax.set_title("RGB Composite by Proper Motion (RA/Dec)", fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    ### ---- Plot 4 ----
    images = []
    for i in range(3):
        filter_pm = (pm>pm_bins[i]) & (pm<pm_bins[i+1]) 
        # Bin
        den, x, y = np.histogram2d(np.array(l_shifted[filter_pm]), np.array(b[filter_pm]), bins=(nra, ndec))
        wn0 = den !=0
        log_den = np.log10(den)
    # Rescale
        p5, p95 = np.percentile(log_den[wn0], [min_count_color, max_count_color])
        img_cur = np.clip(log_den, p5, p95)
        img_cur = (img_cur - p5) / (p95 - p5)
    # Combine
        images.append(img_cur.T)

    # Stack the grayscale images to create an RGB image
    rgb_image = np.dstack((images[0], images[1], images[2]))

    # Plot
    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.imshow(rgb_image, origin='lower', extent=[ra_range_shifted[0], ra_range_shifted[1], dec_range[0], dec_range[1]], aspect='auto', interpolation='bilinear')

    ax.set_xlabel('RA', fontsize=16)
    ax.set_ylabel('Dec', fontsize=16)

    # Axes
    ax.set_xlim(ra_range_shifted)
    ax.set_ylim(dec_range)
    ax.set_xlabel("Galactic Longitude (l) [deg]", fontsize=fontsize_labels)
    ax.set_ylabel("Galactic Latitude (b) [deg]", fontsize=fontsize_labels)
    ax.set_title("RGB Composite by Proper Motion (Galactic Rectangular)", fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()


    ### ---- Plot 5 ----
    # Convert Galactic l to Mollweide format
    l_transformed = (l + 180) % 360 - 180  # Shift to [-180, 180]
    l_transformed = -l_transformed  # Flip to match Mollweide projection
    l_rad = np.radians(l_transformed)
    b_rad = np.radians(b)


    # Proper motion bins for RGB channels
    pm_bins =  np.linspace(pm_cuts[0], pm_cuts[1], 4)  
    nlon, nlat = 360, 180  # Binning resolution

    # Define bins for 2D histogram
    lon_edges = np.linspace(-np.pi, np.pi, nlon + 1)
    lat_edges = np.linspace(-np.pi/2, np.pi/2, nlat + 1)

    # Create RGB channel images
    images = []
    for i in range(3):
        mask = (pm > pm_bins[i]) & (pm < pm_bins[i+1])

        # 2D histogram in Galactic l-b space
        den, _, _ = np.histogram2d(l_rad[mask], b_rad[mask], bins=(lon_edges, lat_edges))

        # Log scaling
        wn0 = den > 0
        den_log = np.zeros_like(den)
        den_log[wn0] = np.log10(den[wn0])

        # Normalize
        p5, p95 = np.nanpercentile(den_log[wn0], [min_count_color,max_count_color])
        img = np.clip(den_log, p5, p95)
        img = (img - p5) / (p95 - p5)

        images.append(img.T)

    # Stack into RGB image
    rgb_image = np.dstack((images[0], images[1], images[2]))

    # Plot Mollweide projection
    fig, ax = plt.subplots(figsize=figsize_moll, subplot_kw={'projection': 'mollweide'})
    mesh = ax.pcolormesh(lon_edges, lat_edges, rgb_image, shading='auto')

    # Add grid and labels
    ax.grid(True, linestyle="dotted", alpha=0.5)

    tick_labels = ["-150°", "-120°", "-90°", "-60°", "-30°", "0°", "30°", "60°", "90°", "120°", "150°"]
    tick_positions = np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=fontsize_ticks)
    ax.tick_params(axis='y', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()


def plot_sky_density_proper_motion3(gaia_data, pm_cuts=[1, 3.5], cmap="inferno"):
    """
    Generates multiple sky density visualizations using HEALPix and standard histograms.
    This version is formatted for publication with unified plotting aesthetics.
    """

    # ------------------------------
    # GLOBAL PLOTTING PARAMETERS
    # ------------------------------
    figsize_wide = (16, 8)
    figsize_moll = (14, 7)
    fontsize_labels = 16
    fontsize_ticks = 14
    fontsize_title = 18
    tick_length = 6
    tick_width = 1.5

    # ------------------------------
    # DATA EXTRACTION
    # ------------------------------
    l = gaia_data['l'].values
    b = gaia_data['b'].values
    ra = gaia_data['ra'].values
    dec = gaia_data['dec'].values
    mag = gaia_data['dered_G'].values
    pm = gaia_data['pm'].values
    dered_G = gaia_data['dered_G'].values
    dered_BP_RP = gaia_data['dered_BP_RP'].values

    ### ---- Plot 1 ----
    ra_range = [0, 360]
    dec_range = np.array([-90, 90])
    rev_rar_range = np.flip(ra_range)
    nra = int(360)
    ndec = int(180)

    fig, ax = plt.subplots(figsize=figsize_wide)

    den, xedges, yedges = np.histogram2d(ra, dec, bins=(nra, ndec))
    den[den == 0] = 1e-6
    den_log = np.log10(den)
    wn0 = den != 0

    ax.pcolormesh(xedges, yedges, den_log.T, cmap=cmap,
                  vmin=np.nanpercentile(den_log[wn0], 10),
                  vmax=np.nanpercentile(den_log[wn0], 90))

    ax.set_xlim(rev_rar_range)
    ax.set_ylim(dec_range)
    ax.set_xlabel('Right Ascension (RA) [deg]', fontsize=fontsize_labels)
    ax.set_ylabel('Declination (Dec) [deg]', fontsize=fontsize_labels)
    ax.set_title('All-Sky Density in Equatorial Coordinates', fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()


    ### ---- Plot 2 ----
    ra_range_shifted = [-180, 180]
    l_shifted = (-l + 180) % 360 - 180
    b_vals = b

    fig, ax = plt.subplots(figsize=figsize_wide)

    den, xedges, yedges = np.histogram2d(l_shifted, b_vals, bins=(nra, ndec), range=[ra_range_shifted, dec_range])
    ax.pcolormesh(xedges, yedges, den.T, cmap=cmap,
                  vmin=np.nanpercentile(den[den > 0], 5),
                  vmax=np.nanpercentile(den[den > 0], 90))

    ax.set_xlim(ra_range_shifted)
    ax.set_ylim(dec_range)
    ax.set_xlabel("Galactic Longitude (l) [deg]", fontsize=fontsize_labels)
    ax.set_ylabel("Galactic Latitude (b) [deg]", fontsize=fontsize_labels)
    ax.set_title("Galactic Density Map (l = 0° Centered)", fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()

    ### ---- Plot 3 ----
    pm_bins = np.linspace(pm_cuts[0], pm_cuts[1], 4)

    images = []
    for i in range(3):
        filter_pm = (pm > pm_bins[i]) & (pm < pm_bins[i + 1])
        den, x, y = np.histogram2d(ra[filter_pm], dec[filter_pm], bins=(nra, ndec))
        wn0 = den != 0
        log_den = np.log10(den)
        p5, p95 = np.percentile(log_den[wn0], [5, 95])
        img_cur = np.clip(log_den, p5, p95)
        img_cur = (img_cur - p5) / (p95 - p5)
        images.append(img_cur.T)

    rgb_image = np.fliplr(np.dstack((images[0], images[1], images[2])))

    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.imshow(rgb_image, origin='lower',
              extent=[rev_rar_range[0], rev_rar_range[1], dec_range[0], dec_range[1]],
              aspect='auto', interpolation='bilinear')

    ax.set_xlim(rev_rar_range)
    ax.set_ylim(dec_range)
    ax.set_xlabel('RA [deg]', fontsize=fontsize_labels)
    ax.set_ylabel('Dec [deg]', fontsize=fontsize_labels)
    ax.set_title("RGB Composite by Proper Motion (RA/Dec)", fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    ### ---- Plot 4 ----
    images = []
    for i in range(3):
        filter_pm = (pm > pm_bins[i]) & (pm < pm_bins[i + 1])
        den, x, y = np.histogram2d(l_shifted[filter_pm], b[filter_pm], bins=(nra, ndec))
        wn0 = den != 0
        log_den = np.log10(den)
        p5, p95 = np.percentile(log_den[wn0], [5, 95])
        img_cur = np.clip(log_den, p5, p95)
        img_cur = (img_cur - p5) / (p95 - p5)
        images.append(img_cur.T)

    rgb_image = np.dstack((images[0], images[1], images[2]))

    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.imshow(rgb_image, origin='lower',
              extent=[ra_range_shifted[0], ra_range_shifted[1], dec_range[0], dec_range[1]],
              aspect='auto', interpolation='bilinear')

    ax.set_xlim(ra_range_shifted)
    ax.set_ylim(dec_range)
    ax.set_xlabel("Galactic Longitude (l) [deg]", fontsize=fontsize_labels)
    ax.set_ylabel("Galactic Latitude (b) [deg]", fontsize=fontsize_labels)
    ax.set_title("RGB Composite by Proper Motion (Galactic Rectangular)", fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()

    ### ---- Plot 5 ----
    l_transformed = (l + 180) % 360 - 180
    l_transformed = -l_transformed
    l_rad = np.radians(l_transformed)
    b_rad = np.radians(b)

    nlon, nlat = 360, 180

    lon_edges = np.linspace(-np.pi, np.pi, nlon + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, nlat + 1)

    images = []
    for i in range(3):
        mask = (pm > pm_bins[i]) & (pm < pm_bins[i + 1])
        den, _, _ = np.histogram2d(l_rad[mask], b_rad[mask], bins=(lon_edges, lat_edges))
        wn0 = den > 0
        den_log = np.zeros_like(den)
        den_log[wn0] = np.log10(den[wn0])
        p5, p95 = np.nanpercentile(den_log[wn0], [5, 99])
        img = np.clip(den_log, p5, p95)
        img = (img - p5) / (p95 - p5)
        images.append(img.T)



    rgb_image = np.dstack((images[0], images[1], images[2]))
    # Set masked/empty pixels to white
    mask = (rgb_image.sum(axis=2) < 0.1)
    rgb_image[mask] = [1.0, 1.0, 1.0]  # RGB for white


    fig, ax = plt.subplots(figsize=figsize_moll, subplot_kw={'projection': 'mollweide'})

        # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    mesh = ax.pcolormesh(lon_edges, lat_edges, rgb_image, shading='auto')

    ax.grid(True, linestyle="dotted", alpha=0.5)
    ax.set_title("RGB Sky Density by Proper Motion (Galactic Mollweide)", fontsize=fontsize_title)

    tick_labels = ["-150°", "-120°", "-90°", "-60°", "-30°", "0°", "30°", "60°", "90°", "120°", "150°"]
    tick_positions = np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=fontsize_ticks)
    ax.tick_params(axis='y', labelsize=fontsize_ticks, length=tick_length, width=tick_width)

    plt.tight_layout()
    plt.show()


def investigation_pipeline(filename, pmra_limits, pmdec_limits, label):
    # ---------------- Load Data ----------------
    path = f"data_unknown/{filename}"
    with fits.open(path) as hdul:
        data = Table(hdul[1].data).to_pandas()
    
    display(data.describe())
    
    pmra_lo, pmra_hi = pmra_limits
    pmdec_lo, pmdec_hi = pmdec_limits

    # ---------------- Initial Plots ----------------
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RA-Dec
    h1 = axes[0].hist2d(data.ra, data.dec, bins=150, cmin=1, cmap="plasma")
    fig.colorbar(h1[3], ax=axes[0], label="Count")
    axes[0].set_xlabel("RA (deg)", fontsize=14)
    axes[0].set_ylabel("Dec (deg)", fontsize=14)
    axes[0].set_title("Spatial Distribution", fontsize=16)

    # Proper Motion
    h2 = axes[1].hist2d(data.pmra, data.pmdec, bins=800, cmin=1, range=[[-50, 50], [-60, 60]], cmap="plasma")
    fig.colorbar(h2[3], ax=axes[1], label="Count")
    axes[1].set_xlabel(r"PMRA  ($\mu_{\alpha}$)", fontsize=14)
    axes[1].set_ylabel(r"PMDEC ($\mu_{\delta}$)", fontsize=14)
    axes[1].set_title("Proper Motion Distribution", fontsize=16)
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-7, 2)
    for line in [pmra_lo, pmra_hi]:
        axes[1].axvline(line, color="red", linestyle="--", linewidth=2)
    for line in [pmdec_lo, pmdec_hi]:
        axes[1].axhline(line, color="red", linestyle="--", linewidth=2)
    axes[1].legend(["Selection Limits"], fontsize=10)

    plt.tight_layout()
    plt.show()

    # ---------------- Dereddening ----------------
    data_dered = reddening_correction(data)

    # ---------------- Proper Motion Filter ----------------
    filtered = data_dered[
        (data_dered.pmra > pmra_lo) & (data_dered.pmra < pmra_hi) &
        (data_dered.pmdec > pmdec_lo) & (data_dered.pmdec < pmdec_hi)
    ]

    # ---------------- Final Plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=600)

    # RA vs Dec
    axes[0].scatter(data.ra, data.dec, c="gray", s=6, alpha=1, label="All Stars")
    axes[0].scatter(filtered.ra, filtered.dec, c="red", s=6, label="Cluster Stars")
    axes[0].set_xlabel("RA [deg]", fontsize=18)
    axes[0].set_ylabel("Dec [deg]", fontsize=18)
    axes[0].legend(fontsize=14, loc="upper left")
    axes[0].set_title(f"RA-Dec Distribution: {label}", fontsize=20)

    # CMD
    axes[1].scatter(data_dered.dered_BP - data_dered.dered_RP, data_dered.dered_G, c="gray", s=6, alpha=1, label="All Stars")
    axes[1].scatter(filtered.dered_BP - filtered.dered_RP, filtered.dered_G, c="red", s=6, label="Cluster Stars")
    axes[1].set_xlabel("BP - RP", fontsize=18)
    axes[1].set_ylabel("Apparent Magnitude", fontsize=18)
    axes[1].legend(fontsize=14, loc="upper left")
    axes[1].set_title(f"CMD: {label}", fontsize=20)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

    return filtered