==========================
Welcome to CLIMADA Petals!
==========================

.. image:: img/CLIMADA_logo_QR.png
   :align: center
   :alt: CLIMADA Logo

CLIMADA stands for CLIMate ADAptation and is a probabilistic natural catastrophe impact model, that also calculates averted damage (benefit) thanks to adaptation measures of any kind (from grey to green infrastructure, behavioural, etc.).

CLIMADA is primarily developed and maintained by the `Weather and Climate Risks Group <https://wcr.ethz.ch/>`_ at `ETH Zürich <https://ethz.ch/en.html>`_.

This is the documentation of the CLIMADA **Petals** module.
Its purpose is generating different types of hazards and more specialized applications than available in the CLIMADA Core module.

.. attention::

   CLIMADA Petals builds on top of CLIMADA Core and is **not** a standalone module.
   Before you start working with Petals, please check out the documentation of the `CLIMADA Core <https://climada-python.readthedocs.io/en/latest/>`_ module, in particular the `installation instructions <https://climada-python.readthedocs.io/en/latest/guide/install.html>`_.

Jump right in:

* :doc:`README <misc/README>`
* `Installation (Core and Petals) <https://climada-python.readthedocs.io/en/latest/guide/install.html>`_
* `GitHub Repository <https://github.com/CLIMADA-project/climada_petals>`_
* :doc:`Module Reference <climada_petals/climada_petals>`

.. ifconfig:: readthedocs

   .. hint::

      ReadTheDocs hosts multiple versions of this documentation.
      Use the drop-down menu on the bottom left to switch versions.
      ``stable`` refers to the most recent release, whereas ``latest`` refers to the latest development version.

.. admonition:: Copyright Notice

   Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in :doc:`AUTHORS.md <misc/AUTHORS>`.

   CLIMADA is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License as published by the Free
   Software Foundation, version 3.

   CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
   PARTICULAR PURPOSE.  See the GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with CLIMADA. If not, see https://www.gnu.org/licenses/.


.. toctree::
   :hidden:

   GitHub Repositories <https://github.com/CLIMADA-project>
   CLIMADA Core <https://climada-python.readthedocs.io/en/latest/>
   Weather and Climate Risks Group <https://wcr.ethz.ch/>


.. toctree::
   :caption: Documentation
   :maxdepth: 1

   glofas_rf
   

.. toctree::
   :caption: API Reference
   :hidden:

   Python Modules <climada_petals/climada_petals>


.. toctree::
   :caption: Tutorials
   :hidden:
   :maxdepth: 2

   Hazard <tutorial/hazard>
   tutorial/climada_engine_SupplyChain
   tutorial/climada_entity_BlackMarble
   tutorial/climada_exposures_openstreetmap
   tutorial/climada_hazard_drought
   Crop Production Risk <tutorial/climada_hazard_entity_Crop>
   Warning Module <tutorial/climada_engine_Warn>


.. toctree::
   :caption: Miscellaneous
   :hidden:
   :maxdepth: 1

   README <misc/README>
   Changelog <misc/CHANGELOG>
   List of Authors <misc/AUTHORS>
   Contribution Guide <https://climada-python.readthedocs.io/en/latest/misc/CONTRIBUTING.html>
   Citation Guide <https://climada-python.readthedocs.io/en/latest/misc/citation.html>
