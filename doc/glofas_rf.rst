==============================================
River Flood Hazards from GloFAS Discharge Data
==============================================

This tutorial will guide you through the GloFAS River Flood module of CLIMADA Petals.
It's purpose is to download river discharge data from the Global Flood Awareness System (GloFAS) and compute flood depths from it.
The data is stored by the `Copernicus Data Store (CDS) <https://cds.climate.copernicus.eu/#!/home>`_ and will be automatically downloaded in the process.

--------
Overview
--------

Because hydrological models for inundation depths are computationally expensive, we do not want to compute flood directly.
Instead, we only use river discharge data and river flood hazard maps.
These hazard maps contain flood footprints for specific return periods.
The idea is to compute equivalent return periods for the discharge data at every pixel and then use the flood hazard maps to compute a flood hazard.
For computing these return periods, we require an extreme value distribution at every point.
In practice, we fit such distributions from the historical discharge data.

Note that computing river flood hazards with this module still is computationally expensive.
We recommend using a powerful machine or even a server cluster for large computations – or likewise a lot of time to spare.

All computations within this module are executed as a Transformation DAG pipeline of the `dantro <https://dantro.readthedocs.io/en/latest/>`_ package.
It handles the data management, caches results for later use and can be controlled via `YAML <https://yaml.org/>`_ configuration files.
We provide one configuration file for the setup (`climada_petals/hazard/rf_glofas/setup.yml`) and one exemplary configuratio file for a river flood foodprint computation task (`climada_petals/hazard/rf_glofas/rf_glofas.yml`).

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

Create Data Directory
^^^^^^^^^^^^^^^^^^^^^

1. Enter the CLIMADA data directory.
   If you did not change the default configuration, this is located in ``$HOME/climada/data``.

  If this directory does not exist, make sure you activated the CLIMADA environment and execute

  .. code-block:: shell

    python -c "import climada"

2. Create a new directory called ``glofas-computation`` in the data directory.

Download Prepared Datasets from the ETH Research Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data required for flood footprint computation has been uploaded to the `ETH Research Collection <https://www.research-collection.ethz.ch/>`_ for your convenience.
You can download the ``gumbel-fit.nc`` and ``flood-maps.nc`` files from there.
Place them in the data directory you created in the last section.

If you downloaded the data and placed it into the ``glofas-computation`` directory, you can directly proceed with `compute`_.
If not, you will have to follow the next steps to compute the input data yourself:

Optional: Create Input Data Using ``setup``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you cannot download the data, you can re-create it on your machine using the :py:func:`climada_petals.hazard.rf_glofas.setup` function.
To do so, follow the optional instructions below.

First, download the global flood hazard map source data.
Freely accessible global flood hazard maps are available from the Joint Research Centre (JRC) Data Catalogue as GeoTIFF files: https://data.jrc.ec.europa.eu/collection/id-0054

1. Download all "Flood hazard map of the World" from the JRC Data Catalogue.
   Click on each respective entry, navigate to the "Data access" section and click "Download".
2. Unzip the downloaded files.
3. Copy or move the unzipped folders into the data directory.

Now we will execute the ``setup`` function.
It will run two data transformation tasks.
Both tasks are defined in the ``climada_petals/hazard/rf_glofas/setup.yml`` configuration file.

* Task 1: `glofas_historical_fits`

  1. Download historical river discharge data from 1979 to 2021.
  2. Compute the yearly maximum for each grid cell.
  3. Fit a right-skewed Gumbel distribution to the resulting time series of every cell.
  4. Store the fit parameters ``loc`` and ``scale`` as data variables in a NetCDF file.

  .. literalinclude:: /../climada_petals/hazard/rf_glofas/setup.yml
    :lines: 62-94
    :lineno-match:
    :caption: setup.yml

* Task 2: `flood_maps_merge`

  1. Load the flood hazard map GeoTIFF files.
  2. Merge them into a single dataset with ``return_period`` as new dimension.
  3. Store the inundation depth as data variable in a NetCDF file.

  .. literalinclude:: /../climada_petals/hazard/rf_glofas/setup.yml
    :lines: 96-111
    :lineno-match:
    :caption: setup.yml

Task 1 is computationally intensive and might take some time.
It is best to run this function on a capable private computer or a server cluster, if available.

By default, the function will execute using the multithreading capabilities of the system you run on.
Alternatively, you can use the :py:func:`climada_petals.hazard.rf_glofas.dask_client` context manager.
It will instantiate a ``dask.distributed.Client``, which creates a ``dask.distributed.LocalCluster`` to exploit multi-processing for computing the tasks.
The settings for the cluster strongly depend on the machine you run the code on, and on the workload.

.. code-block:: python

  from climada_petals.hazard.rf_glofas import setup, dask_client

  # Option 1: Execute `setup` with default multithreading
  setup()

  # Option 2: Execute `setup` in parallel.
  # 4 processes, each with 2 threads and a memory budget of 4GB
  with dask_client(n_workers=4, threads_per_worker=2, memory_limit="8G"):
      setup()

The ``setup`` function actually is just a wrapper around :py:func:`climada_petals.hazard.rf_glofas.dantro_transform` with a defaulted configuration path and no return value.

.. _compute:

---------------------------
Computing a Flood Footprint
---------------------------

With the required data in place and access to the Copernicus Data Store, we can proceed to compute a flood footprint.
In the end, we want to arrive at a ``Hazard`` object we can use for computations in CLIMADA.

The overall procedure is as follows:

1. Copy the file ``climada_petals/hazard/rf_glofas/rf_glofas.yml`` into another directory.
   It contains the default configuration for the hazard computation pipeline.
2. Adjust the configuration file to fit your use case.
3. Execute the ``compute_hazard_series`` function with the adjusted configuration file as argument.

The Pipeline Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    In order to understand everything that is going on in the pipeline and its configuration file, you will have to take a look at the `dantro documentation <https://dantro.readthedocs.io/en/latest/index.html>`_, in particular:

    * `Data Manager <https://dantro.readthedocs.io/en/latest/data_io/data_mngr.html>`_
    * `Data Processing <https://dantro.readthedocs.io/en/latest/data_io/data_ops.html>`_
    * `Data Transformation Framework <https://dantro.readthedocs.io/en/latest/data_io/transform.html>`_

First, we'll have a detailed look at the default configuration file: ``climada_petals/hazard/rf_glofas/rf_glofas.yml``
Open it with a text editor.

The first entry ``data_dir`` is commented out.
In that case, the directory ``glofas-computation`` in the CLIMADA data directory will be used.
Where this directory is located depends on the `configuration <https://climada-python.readthedocs.io/en/latest/guide/Guide_Configuration.html#2.A.--Configuration-files>`_ of your CLIMADA setup.
The default location is ``~/climada/data/glofas-computation``:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 3-5
    :emphasize-lines: 3
    :lineno-match:
    :caption: rf_glofas.yml

The second entry is the ``data_manager`` tree.
It specifies which data is to be loaded before the transformation pipeline commences and where the output is stored.
The ``load_cfg`` simply loads the data created in the previous section.
You need not change these settings:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 8-24
    :lineno-match:
    :caption: rf_glofas.yml

The third entry is the ``plot_manager`` tree.
We are "hijacking" the dantro Plot Manager for computing the transformation.
Instead of a plot function at the end of the pipeline, we call a ``finalize`` function that stores the result in the Data Manager or in a file, depending on our settings in the transformation configuration.
The output directory is set with ``out_dir`` and will be a subdirectory of the ``data_manager: out_dir`` whose name is the time stamp at which the transformation pipeline was started.
You need not change these settings:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 26-31
    :emphasize-lines: 3
    :lineno-match:
    :caption: rf_glofas.yml

The last entry is the ``eval`` tree and specifies the evaluation tasks and their respective transformation pipelines.
It can define multiple tasks, as in case of ``setup.yml``, but for computing a hazard we should only need one.
First, we define some defaults for each task (this is technically not needed because we only use a single task here, but it declutters the actual task a bit).
The ``plot_func`` is the function receiving the data resulting from the transformation pipeline.
Again, we do not use this for plotting, but for our own purposes.
In this case, we always call the :py:func:`climada_petals.hazard.rf_glofas.transform_ops.finalize` function, which writes datasets to files or stores them in the dantro ``DataManager``, depending on (keyword) arguments present in the task (see below):

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 38-43
    :emphasize-lines: 6
    :lineno-match:
    :caption: rf_glofas.yml

The other default config sets the default settings for the `dantro file cache <https://dantro.readthedocs.io/en/latest/data_io/transform.html#the-file-cache>`_.
It can write the result of every single transformation to a file and retrieve it if the same set of operations that created the cache file is executed again.
This is based on the exact configuration that created the cache file in the first place and therefore safe against changes to arguments of the transform operations.
It is obviously very useful – we do not need to compute all nodes of the transform DAG again if only a single operation was changed.

.. note::

  We explicitly set ``netcdf4`` as the engine/backend for reading and writing cache files because it is the only backend that supports writing files in the multi-processing environment of ``dask.distributed``.
  See the `xarray docs <https://docs.xarray.dev/en/stable/user-guide/dask.html#reading-and-writing-data>`_ for further information.

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 52-65
    :lineno-match:
    :caption: rf_glofas.yml

Next, we define our actual evaluation task for computing the flood hazard, and we base it on the default settings given above:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 67-71
    :lineno-match:
    :caption: rf_glofas.yml

Before the transformation, we select data from the data manager and give them specific tags to use them in the transform pipeline:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 73-75
    :lineno-match:
    :caption: rf_glofas.yml

Now, we finally specify the actual data transformation.
The first operation calls :py:func:`climada_petals.hazard.rf_glofas.transform_ops.download_glofas_discharge` to download the forecast data from the CDS.

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 77-83
    :lineno-match:
    :caption: rf_glofas.yml

The second operation computes the maximum over multiple slices along the time ``step`` dimension.
Here, we compute the maximum over the first 3 and the first 5 days of the forecast.
See :py:func:`climada_petals.hazard.rf_glofas.transform_ops.max_from_isel` for details.
This operation reduces over the ``step`` dimension, and adds a new ``select`` dimension with integer values.

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 85-92
    :lineno-match:
    :caption: rf_glofas.yml

Next, we select the ``loc`` and ``scale`` arrays from the ``gumbel-fit.nc`` dataset.

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 94-98
    :lineno-match:
    :caption: rf_glofas.yml

Then, we compute the return period with :py:func:`climada_petals.hazard.rf_glofas.transform_ops.return_period`.
Here, we add particular settings for the file cache.
We always want a cache file to be written (except when the file already exists), and we always want it to be read in again before continuing to the next operation.
We use this feature to load the file with new chunking.
This is important because chunking has significant performance implications.
In the next operation, we will interpolate the data in space, which benefits significantly from chunks which span the entire latitude/longitude grid (the interpolated data fits into a single chunk).

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 100-112
    :lineno-match:
    :caption: rf_glofas.yml

Next, we interpolate in space using :py:func:`climada_petals.hazard.rf_glofas.transform_ops.interpolate_space`.
Again, we use the cache read-write mechanism of dantro to reload the result with different chunking.
In this case, we want to fit all data at a single lat/lon coordinate into one chunk, and still have as few chunks as reasonably possible.

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 114-126
    :lineno-match:
    :caption: rf_glofas.yml

Finally, we compute the flood depth from the interpolated data with :py:func:`climada_petals.hazard.rf_glofas.transform_ops.flood_depth`.

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 129-131
    :lineno-match:
    :caption: rf_glofas.yml

To store the data in a dantro ``DataManager`` instance, we pass said data manager as DAG node as well:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 133-135
    :lineno-match:
    :caption: rf_glofas.yml

All other keywords that are not recognized by dantro will be passed to the ``plot_func`` :py:func:`climada_petals.hazard.rf_glofas.transform_ops.finalize`.
We want the final result ``flood_depth`` to be stored in the data manager so we can later retrieve it in the code.
Therefore, we add:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 137-139
    :lineno-match:
    :caption: rf_glofas.yml

Adjusting the Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that you are familiar with the configuration, copy the file ``rf_glofas.yml`` to another location and optionally rename it (e.g., to ``my_cfg.yml``).
You can now adjust the ``transform`` to fit your use case.
The most common changes will be to the data downloaded from the CDS, in particular w.r.t. the date parameters, the ``country``, and the ``leadtime_hour``:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 78-83
    :lineno-match:
    :emphasize-lines: 2,4,5
    :caption: my_cfg.yml

Depending on the ``leadtime_hour`` and your particular use case, you might then want to adjust the time ranges over which to compute the maxima:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 85-92
    :lineno-match:
    :emphasize-lines: 5-7
    :caption: my_cfg.yml

Then you can finally execute the pipeline!

Executing the Pipeline
^^^^^^^^^^^^^^^^^^^^^^

The pipeline is executed with :py:func:`climada_petals.hazard.rf_glofas.dantro_transform`.
The first argument of this function is the location of the pipeline configuration file.
Pass the path to your adjusted configuration, or use the default value, which is the default file ``rf_glofas.yml``.
Like with :py:func:`climada_petals.hazard.rf_glofas.setup`, additional keyword arguments will be passed to the ``dask.distributed.Client`` constructor.

.. code-block:: python

  from pathlib import Path
  from climada_petals.hazard.rf_glofas import dantro_transform, dask_client

  # Default pipeline
  data_manager = dantro_transform()

  # Custom pipeline with multi-processing
  with dask_client(4, 2, "4G"):
      data_manager = dantro_transform(Path("~/my_cfg.yml").expanduser())

``dantro_transform`` returns the ``DataManager`` for the pipeline.
If you stored the flood depth via the ``to_dm`` node in the configuration file, this data manager will contain the dataset we are looking for to build ``Hazard`` objects.

Creating Hazards from the Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the computation successful, we now want to create `Hazard <https://climada-python.readthedocs.io/en/stable/tutorial/climada_hazard_Hazard.html>`_ objects.
The resulting data is usually multi-dimensional, which is why we typically create multiple Hazard objects from it.
Two obvious dimensions are the spatial ones, longitude and latitude.
Ignoring these (as they must persist into the ``Hazard`` object), we can decide on one more dimension to merge into a single hazard.

Given the default configuration ``rf_glofas.yml``, there are three more dimensions:
The number of the forecast ensemble member (``number``), the date at which the forecast was issued (``date``), and the selection dimension from ``max_from_isel`` (``select``).
A straightforward way of merging each event is by the ensemble dimension (``number``).
Then we receive a ``Hazard`` for every combination of ``date`` and ``select``, containing as many events as ``number`` has coordinates (the size of the ensemble in this case).

The task of splitting and concatenating along particular dimensions of the dataset and creating Hazard objects is performed by :py:func:`climada_petals.hazard.rf_glofas.transform_ops.hazard_series_from_dataset`.
We put in the data as file path or xarray ``Dataset`` and receive a `pandas.Series <https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#series>`_ with the hazard objects as values and the remaining dimension coordinates as `MultiIndex <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`_.
The dimension name which is to be considered the event dimension in a ``Hazard`` instance must be specified as the ``event_dim`` argument.

.. tip::

    You can, but need not, execute ``hazard_series_from_dataset`` within the same dask client context.

.. code-block:: python

  from climada_petals.hazard.rf_glofas import (
      dask_client,
      dantro_transform,
      hazard_series_from_dataset,
  )

  # Default pipeline
  with dask_client(4, 2, "4G"):
      data_manager = dantro_transform()

      # Pass dataset in data manager container to function
      hazards = hazard_series_from_dataset(
          data_manager["flood_depth"].data, event_dim="number"
      )

  # Select the Hazard object for the 5-day-maximum of the forecast on 2022-08-11
  hazard = hazards[1, "2022-08-11"]

Great! We have a ``Hazard``!

.. note::

    The actual ``Hazard`` instance returned is a subclass of ``Hazard``, :py:class:`climada_petals.hazard.river_flood.RiverFlood`.

Handling Very Large Data
^^^^^^^^^^^^^^^^^^^^^^^^

In case of very large amounts of data, it might be suitable to *not* pass the dataset to the data manager but only write it into a file.
For this, add a ``to_file`` node to the pipeline configuration, as present in the ``setup.yml``, e.g.:

.. code-block:: yaml

  eval:
    with_cache:
      to_file:
        - flood_maps

.. hint::

    See :py:func:`climada_petals.hazard.rf_glofas.transform_ops.finalize` for information on how to specify ``to_file``.

Then you can read this file again when building hazard objects.
Note that you might want to customize the ``out_dir`` of both the ``data_manager`` and the ``plot_manager`` nodes in this case:

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 8-9
    :lineno-match:
    :emphasize-lines: 2
    :caption: my_cfg.yml

.. literalinclude:: /../climada_petals/hazard/rf_glofas/rf_glofas.yml
    :lines: 26-28
    :lineno-match:
    :emphasize-lines: 3
    :caption: my_cfg.yml

The function :py:func:`climada_petals.hazard.rf_glofas.transform_ops.hazard_series_from_dataset` also accepts a file path as ``data`` parameter:

.. code-block:: python

  from climada_petals.hazard.rf_glofas import (
      dantro_transform,
      hazard_series_from_dataset,
  )

  # Custom pipeline might write data into a file using 'to_file'
  dantro_transform(Path("~/my_cfg.yml").expanduser())
  hazards = hazard_series_from_dataset("my_output_file.nc", event_dim="number")
