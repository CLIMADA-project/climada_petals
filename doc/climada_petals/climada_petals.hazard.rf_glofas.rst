===================================================
River Flood from GloFAS River Discharge Data Module
===================================================

-----------
Main Module
-----------

.. automodule:: climada_petals.hazard.rf_glofas.river_flood_computation
    :members:
    :undoc-members:
    :show-inheritance:

-------------------------
Transformation Operations
-------------------------

.. automodule:: climada_petals.hazard.rf_glofas.transform_ops
    :members:
    :undoc-members:
    :show-inheritance:

----------------
Helper Functions
----------------

These are the functions exposed by the module.

.. automodule:: climada_petals.hazard.rf_glofas.rf_glofas
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: climada_petals.hazard.rf_glofas.setup
    :members:
    :undoc-members:
    :show-inheritance:

---------------------
CDS Glofas Downloader
---------------------

.. autofunction:: climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request

The default configuration for each product will be updated with the ``request_kw`` from :py:func:`climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request`:

.. autodata:: climada_petals.hazard.rf_glofas.cds_glofas_downloader.DEFAULT_REQUESTS
