# Data Aquisition

- During this project, data was aquired from the Gaia ESA Archive using ADQL queries, in this file I simply outline the files this project relies on and what ADQL queries were used to obtain this data. The justifications of these search criteria and provided throughout the project and report.
- This is provided to allow the analysis in this notebook to be reproduced
- A skeleton function is provided in `GA_Analysis` for preforming these Queries directly using `Astroquery`.

## Explaination of Syntax 
- the following filters are applied below in various queries, I hope to give an brief overview here.

| Filter | Condition | Purpose / Explanation |
|---|---|---|
| Magnitude cut | `gs.phot_g_mean_mag BETWEEN 10 AND 20.5` |  Stars within a Gaia G-band (brightness) range. Excludes very bright (close) or very faint (uncertain) stars. |
| Parallax cut | `gs.parallax < 0.1` or `gs.parallax BETWEEN -0.1 AND 0.1` | Removed nearby stars ie those with large parallaxes and keeps distant or halo-like population stars. I allow for small negative parallaxes due to measurement uncertainty found in gaia. |
| RUWE cut | `gs.ruwe < 1.4` | This ensures high-quality astrometric solutions. Values above 1.4 typically are a flag for unreliable or blended sources. |
| Proper motion cut | `(gs.pmra * gs.pmra + gs.pmdec * gs.pmdec) < 16` or `< 144` | Filters out stars with high sky motion and hence close. Equivalent to proper motion < 4 mas/yr or 12 mas/yr depending on threshold. Retains slower-moving, distant stars (e.g., in streams or halo). |
| Radial velocity check | `gs.radial_velocity IS NOT NULL` | This ensures the stars have a measured radial velocity (used for kinematic analysis with galpy). |
| Distance availability | `gd.r_med_photogeo IS NOT NULL` | Keeps only stars with a valid photogeometric distance estimate (from Bailer-Jones). |
| Random downsampling | `gs.random_index BETWEEN 0 AND 700000000` | Limits the number of sources returned, to stop the file being too large using gaias random index |
| Cone search (spatial filter) | `1 = CONTAINS(POINT('ICRS', gs.l, gs.b), CIRCLE('ICRS', x, y, r))` | Selects stars within a circular region (e.g., around a cluster). |


---

## `M3` (Investigation to inform later cuts - from the same random subsample) 
- This file was used as an investigation into the quality of cuts chosen on later on in the investigation to verify that the desired population would be visible 

```sql
SELECT
  gs.source_id,
  gs.l, gs.b, gs.ra, gs.dec,
  gs.ra_error, gs.dec_error,
  gs.phot_g_mean_mag,
  gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
  gs.parallax, gs.parallax_error,
  gs.pmra, gs.pmdec, gs.pmra_error, gs.pmdec_error,
  gs.radial_velocity,
  gs.bp_rp,
  gd.r_med_photogeo, gd.r_hi_photogeo, gd.r_lo_photogeo,
  gs.logg_gspphot, gs.teff_gspphot
FROM gaiaedr3.gaia_source AS gs
LEFT JOIN external.gaiaedr3_distance AS gd
  ON gs.source_id = gd.source_id
WHERE 
.  gs.phot_g_mean_mag BETWEEN 10 AND 20.5 AND
  gs.parallax < 0.1 AND
  gs.ruwe < 1.4 AND
  (gs.pmra * gs.pmra + gs.pmdec * gs.pmdec) < 16 AND
  gd.r_med_photogeo IS NOT NULL AND
  1 = CONTAINS(
    POINT('ICRS', gs.l, gs.b),
    CIRCLE('ICRS', 42.21695, 78.70685, 1)
  ) AND
  gs.random_index BETWEEN 0 AND 700000000;
  ```

  ## `NGC1851` (Investigation to inform later cuts  - from the same random subsample) 
  
  ```sql
  SELECT 
  gs.source_id, gs.l, gs.b, gs.ra, gs.dec,
  gs.ra_error, gs.dec_error,
  gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
  gs.parallax, gs.parallax_error,
  gs.pmra, gs.pmdec, gs.pmra_error, gs.pmdec_error,
  gs.radial_velocity, gs.bp_rp,
  gd.r_med_photogeo, gd.r_hi_photogeo, gd.r_lo_photogeo,
  gs.logg_gspphot, gs.teff_gspphot
FROM gaiaedr3.gaia_source AS gs
LEFT JOIN external.gaiaedr3_distance AS gd
  ON gs.source_id = gd.source_id
WHERE
  gs.phot_g_mean_mag BETWEEN 10 AND 20.5 AND
  gs.parallax < 0.1 AND
  gs.ruwe < 1.4 AND
  (gs.pmra * gs.pmra + gs.pmdec * gs.pmdec) < 16 AND
  gd.r_med_photogeo IS NOT NULL AND
  1 = CONTAINS(
    POINT('ICRS', gs.l, gs.b),
    CIRCLE('ICRS', 244.51323, -35.03598, 1)
  ) AND
  gs.random_index BETWEEN 0 AND 700000000;
  ```

## `Allsky_Gaia_42481846` - Low Propermotion Cut (<4)


```sql
SELECT 
  gs.source_id, gs.l, gs.b, gs.ra, gs.dec,
  gs.ra_error, gs.dec_error,
  gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
  gs.parallax, gs.pmra, gs.pmdec, gs.pmra_error, gs.pmdec_error,
  gd.r_med_photogeo, gd.r_hi_photogeo, gd.r_lo_photogeo
FROM gaiaedr3.gaia_source AS gs
LEFT JOIN external.gaiaedr3_distance AS gd
  ON gs.source_id = gd.source_id
WHERE
  gs.phot_g_mean_mag BETWEEN 10 AND 20 AND
  gs.parallax BETWEEN -0.1 AND 0.1 AND
  gs.ruwe < 1.4 AND
  (gs.pmra * gs.pmra + gs.pmdec * gs.pmdec) < 16;
```
## `Allsky_Gaia_4559940` - Low Propermotion Cut (<12)

```sql
SELECT 
  gs.source_id, gs.l, gs.b, gs.ra, gs.dec,
  gs.ra_error, gs.dec_error,
  gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
  gs.parallax, gs.parallax_error,
  gs.pmra, gs.pmdec, gs.pmra_error, gs.pmdec_error,
  gs.radial_velocity, gs.bp_rp,
  gd.r_med_photogeo, gd.r_hi_photogeo, gd.r_lo_photogeo
FROM gaiaedr3.gaia_source AS gs
LEFT JOIN external.gaiaedr3_distance AS gd
  ON gs.source_id = gd.source_id
WHERE
  gs.phot_g_mean_mag BETWEEN 10 AND 20.5 AND
  gs.parallax BETWEEN -0.1 AND 0.1 AND
  gs.ruwe < 1.4 AND
  (gs.pmra * gs.pmra + gs.pmdec * gs.pmdec) < 144 AND
  gd.r_med_photogeo IS NOT NULL AND
  gs.random_index BETWEEN 0 AND 700000000;
```


## `Allsky_Gaia_394217` - Sample with Radial Velocities

```sql
SELECT 
  gs.source_id, gs.l, gs.b, gs.ra, gs.dec,
  gs.ra_error, gs.dec_error,
  gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
  gs.parallax, gs.parallax_error,
  gs.pmra, gs.pmdec, gs.pmra_error, gs.pmdec_error,
  gs.radial_velocity, gs.bp_rp,
  gd.r_med_photogeo, gd.r_hi_photogeo, gd.r_lo_photogeo,
  gs.logg_gspphot, gs.teff_gspphot
FROM gaiaedr3.gaia_source AS gs
LEFT JOIN external.gaiaedr3_distance AS gd
  ON gs.source_id = gd.source_id
WHERE
  gs.phot_g_mean_mag BETWEEN 10 AND 20 AND
  gs.parallax BETWEEN -0.1 AND 0.1 AND
  gs.ruwe < 1.4 AND
  (gs.pmra * gs.pmra + gs.pmdec * gs.pmdec) < 16 AND
  gs.radial_velocity IS NOT NULL AND
  gd.r_med_photogeo IS NOT NULL;
```

--- 
