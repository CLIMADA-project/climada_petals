==============================================
River Flood Hazards from GloFAS Discharge Data
==============================================

This tutorial will guide you through the GloFAS River Flood module of CLIMADA Petals.
It's purpose is to download river discharge data from the Global Flood Awareness System (GloFAS) and compute flood depths from it.
The data is stored by the `Copernicus Data Store (CDS) <https://cds.climate.copernicus.eu/#!/home>`_ and will be automatically downloaded in the process.

--------
Overview
--------

Instead of employing a computationally expensive hydrological model to compute inundation depths, this module uses a simplified statistical approach to compute flooded areas.
As an input, the approach uses river discharge data and river flood hazard maps.
These hazard maps contain flood footprints for specific return periods.
The idea is to compute equivalent return periods for the discharge data at every pixel and then use the flood hazard maps to compute a flood hazard.
For computing these return periods, we require an extreme value distribution at every point.
In practice, we fit such distributions from the historical discharge data.

Depending on your area and time series of interest the computational cost and the amount of data produced can be immense.
For a larger country, however, a single flood inundation footprint can be computed within few minutes on a decently modern machine.

------------
Preparations
------------

We need to prepare three things: The flood hazard maps, the extreme value distributions, and access to the CDS API.

Copernicus Data Store API Access

1. Register at the `Copernicus Data Store (CDS) <https://cds.climate.copernicus.eu/#!/home>`_ and log in.
2. Check out the `CDS API HowTo <https://cds.climate.copernicus.eu/api-how-to>`_.
   In the section "Install the CDS API key", copy the content of the black box on the right.
3. Create a file called ``.cdsapirc`` in your home directory and paste the contents of the black box into it.

   If you are unsure where to put the file and you are working on a Linux or macOS system, open a terminal and execute

   .. code-block:: shell

      cd $HOME
      touch .cdsapirc

   Now the file is created and can be opened with your favorite text editor.

Use Prepared Datasets
^^^^^^^^^^^^^^^^^^^^^

The Gumbel distribution fit parameter data has been uploaded to the `ETH Research Collection <https://www.research-collection.ethz.ch/>`_ for your convenience: `Gumbel distribution fit parameters for historical GloFAS river discharge data (1979–2015) <https://doi.org/10.3929/ethz-b-000641667>`_

This dataset and the global flood hazard maps will be automatically downloaded when executing

.. code-block:: python

    from climada_petals.hazard.rf_glofas import setup_all

    setup_all()

Alternatively, you can download the data yourself or specify custom paths to datasets on
your machine.

After this step, you should have the following files in your ``<climada-dir>/data/river-flood-computation``:

* ``gumbel-fit.nc``: A NetCDF file containing ``loc``, ``scale`` and ``samples`` variables with dimensions ``latitude`` and ``longitude`` on a grid matching the input discharge data (here: GloFAS).
* ``flood-maps.nc``: A NetCDF file giving the ``flood_depth`` with dimensions ``latitude``, ``longitude``, and ``return_period``. The grid is allowed to differ from that of the discharge and the Gumbel fit parameters and is expected to have a higher resolution.
* ``FLOPROS_shp_V1/FLOPROS_shp_V1.shp``: A shapefile containing flood protection standards for the entire world, encoded as return period against which the local measures are protecting against. 

.. _compute:

---------------------------
Computing a Flood Footprint
---------------------------

With the required data in place and access to the Copernicus Data Store, we can proceed to compute a flood footprint.
In the end, we want to arrive at a ``Hazard`` object we can use for computations in CLIMADA.

The overall procedure is as follows:

1. Instantiate an object of :py:class:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation`.
2. Use it to download discharge data (either ensemble forecasts or historical reanalysis) from the CDS.
3. Compute flood inundation footprints from the downloaded data with :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.compute`.
4. Create a series of hazard objects (or a single object) from the data using :py:func:`~climada_petals.hazard.rf_glofas.rf_glofas.hazard_series_from_dataset`.

.. code-block:: python

    from climada_petals.hazard.rf_glofas import (
        RiverFloodInundation,
        hazard_series_from_dataset,
    )

    forecast_date = "2023-08-01"
    rf = RiverFloodInundation()
    rf.download_forecast(
        countries="Switzerland",
        forecast_date=forecast_date,
        lead_time_days=5,
        preprocess=lambda x: x.max(dim="step"),
    )
    ds_flood = rf.compute()
    hazard = hazard_series_from_dataset(ds_flood, "flood_depth", "number")

Step-By-Step Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.compute` method is a shortcut for the steps of the flood model algorithm that compute flood depth from the discharge input.

The single steps are as follows:

#. Computing the return period from the input discharge with :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.return_period`.
   To that end, the fitted Gumbel distributions are used and a return period is computed by :math:`r(q) = (1 - \text{cdf}(q))^{-1}`, where :math:`\text{cdf}` is the cumulative distribution function of the fitted Gumbel distribution and :math:`q` is the input discharge.

   .. code-block:: python

        discharge = rf.download_forecast(
            countries="Switzerland",
            forecast_date=forecast_date,
            lead_time_days=5,
            preprocess=lambda x: x.max(dim="step"),
        )
        return_period = rf.return_period(discharge)

   Alternatively, bootstrap sampling can be employed to represent the statistical uncertainty in the return period computation with :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.return_period_resample`.
   In bootstrap sampling, we draw random samples from the fitted Gumbel distribution and fit a new distribution from them.
   This process can be repeated an arbitrary number of times.
   The resulting distribution quantifies the uncertainty in the original fit.
   The first argument to the method is the number of samples to draw while bootstrapping (i.e., how many samples the resulting distribution should have).

   .. code-block:: python

        return_period = rf.return_period_resample(10, discharge)

#. Regridding the return period onto the higher resolution grid of the flood hazard maps with :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.regrid`:

   .. code-block:: python

        return_period_regrid = rf.regrid(return_period)

#. *Optional:* Applying the protection level with :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.apply_protection`:

   .. code-block:: python

        return_period_regrid_protect = rf.apply_protection(return_period_regrid)

#. Computing the flood depth from the regridded return period by interpolating between flood hazard maps for various return periods with :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.flood_depth`

   .. code-block:: python

        flood_depth = rf.flood_depth(return_period_regrid)
        flood_depth_protect = rf.flood_depth(return_period_regrid_protect)

   If :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.compute` was executed with ``apply_protection="both"`` (default), it will merge the data arrays for flood depth without protection applied and with protection applied, respectively, into a single dataset and return it.

Passing Keyword-Arguments to ``compute``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to pass custom arguments to the methods called by :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.compute` without calling each method individually, you can do so via the ``resample_kws`` and ``regrid_kws`` arguments.

If you add ``resample_kws``, :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.compute` will call :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.return_period_resample` instead of :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.return_period` and pass the mapping as keyword arguments.

Likewise, ``regrid_kws`` will be passed as keyword arguments to :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.regrid`.

.. code-block:: python

    ds_flood = rf.compute(
        resample_kws=dict(num_bootstrap_samples=20, num_workers=4),
        regrid_kws=dict(reuse_regridder=True)
    )

Creating Hazards from the Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the computation successful, we now want to create `Hazard <https://climada-python.readthedocs.io/en/stable/tutorial/climada_hazard_Hazard.html>`_ objects.
The resulting data is usually multi-dimensional, which is why we typically create multiple Hazard objects from it.
Two obvious dimensions are the spatial ones, longitude and latitude.
Ignoring these (as they must persist into the ``Hazard`` object), we can decide on one more dimension to merge into a single hazard.

If we use an ensemble forecast like in the above example, and decide *not* to compute the maximum in time, the dataset has four coodinates: ``latitude``, ``longitude``, ``step``, and ``number``, with the latter two indicating the lead time step and the ensemble member, respectively.
Employing bootstrap sampling would add another dimension ``sample``.
To create hazard objects, we would have to decide which of these dimensions should encode the "event" dimension in the ``Hazard``.
For each combination of the remaining dimension coordinates, a new Hazard object would then be created.

The task of splitting and concatenating along particular dimensions of the dataset and creating Hazard objects is performed by :py:func:`climada_petals.hazard.rf_glofas.rf_glofas.hazard_series_from_dataset`.
We put in the data as file path or xarray ``Dataset`` and receive a `pandas.Series <https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#series>`_ with the hazard objects as values and the remaining dimension coordinates as `MultiIndex <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`_.
The dimension name which is to be considered the event dimension in a ``Hazard`` instance must be specified as the ``event_dim`` argument.

.. tip::

    If the dataset is three-dimensional, :py:func:`climada_petals.hazard.rf_glofas.rf_glofas.hazard_series_from_dataset` will return a single Hazard object instead of a ``pandas.Series``.

.. code-block:: python

    discharge = rf.download_forecast(
        countries="Switzerland",
        forecast_date="2023-08-01",
        lead_time_days=5,
    )

    # Compute flood for maximum over lead time
    ds_flood = rf.compute(discharge.max(dim="step"))

    # Single hazard return (no remaining dimensions)
    hazard = hazard_series_from_dataset(ds_flood, "flood_depth", "number")

    # Compute flood for each lead time day *and* bootstrap sample
    ds_flood_multidim = rf.compute(discharge, resample_kws=dict(num_bootstrap_samples=20))

    # Series with MultiIndex: step, member
    # Each hazard with 20 events (samples)
    hazard_series = hazard_series_from_dataset(ds_flood, "flood_depth", "sample")


Storing Data
^^^^^^^^^^^^

Use :py:func:`climada_petals.hazard.rf_glofas.transform_ops.save_file` to store xarray Datasets or DataArrays conveniently.

.. tip::

    Storing your result is important for two reasons:

    #. Computing flood footprints for larger areas or multiple events can take a lot of time.
    #. Loading flood footprints into ``Hazard`` objects requires transpositions that do not commute well with the lazy computations and evaluations by xarray.
       Storing the data and re-loading it before plugging it into :py:func:`~climada_petals.hazard.rf_glofas.rf_glofas.hazard_series_from_dataset` will likely increase performance.

By default, data is stored without compression and encoded in 32-bit floats.
This maintains a reasonable accuracy while reducing file size by half even though no compression is applied.
Compression will drastically reduce the storage space needed for the data.
However, it also creates a heavy burden on the CPU and especially multiprocessing and multithreading tasks suffer heavily.
If storage space permits, it is therefore recommended to store the data without compression.

.. warning::  Saving results of computations **with** compression is **not** recommended, because performance might be impeded **a lot**!

To enable compression, add ``zlib=True`` as argument to :py:func:`~climada_petals.hazard.rf_glofas.transform_ops.save_file`.
The default compression level is ``complevel=4``.
The compression level may range from 1 to 9.

Because storing without compression does not compromise multiprocessing performance, it might be feasible to first write *without* compression after computing the result, and then to re-write *with* compression separately to save storage space.
The reason for this is that xarray uses dask to perform computations lazily.
Only when data is required, dask will compute it according to the transformations applied on the data.
This does not commute well with compression.

The following code will likely run much faster than directly writing ``ds_flood`` with compression, especially when the data is large.
However, it requires the space to once store the entire dataset without compression.

.. code-block:: python

    from pathlib import Path
    import xarray as xr
    from climada_petals.hazard.rf_glofas import save_file

    rf.download_forecast(
        countries="Switzerland",
        forecast_date="2023-08-01",
        lead_time_days=5,
    )
    ds_flood = rf.compute()

    # Save without compression (default)
    outpath = Path("out.nc")
    save_file(ds_flood, outpath)
    ds_flood.close()  # Release data

    # Re-open, and save with compression into "out-comp.nc"
    with xr.open_dataset(outpath, chunks="auto") as ds:
        save_file(ds, outpath.with_stem(outpath.stem + "-comp"), zlib=True)

    # Delete initial file
    outpath.unlink()

Storing Intermediate Data
^^^^^^^^^^^^^^^^^^^^^^^^^

By default, :py:class:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation` stores the result of each computation step in a cache directory and reloads it for the next step.
The reason for this is similar to the issue with compression:
To perform our computations, the data has to be transposed often.
Multiple transpositions of a dataset in memory are costly, but storing data and reopening it transposed is fast.
Especially for larger data that do not fit into memory at once, :py:attr:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.store_intermediates` should therefore be set to ``True`` (default).

The intermediate data is stored in a cache directory which is deleted after the :py:class:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation` instance is closed or deleted.
While it exists, the cached data can be accessed via the :py:attr:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.cache_paths` after the computation:

.. code-block:: python

    import xarray as xr

    rf.download_forecast(
        countries="Switzerland",
        forecast_date="2023-08-01",
        lead_time_days=5,
    )
    ds_flood = rf.compute()

    # Plot regridded return period
    with xr.open_dataarray(rf.cache_paths.return_period_regrid, chunks="auto") as da_rp:
        da_rp.isel(step=0).max(dim="member").plot()
