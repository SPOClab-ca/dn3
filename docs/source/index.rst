.. Deep Neurophysiology Toolbox (DNPT) documentation master file, created by
   sphinx-quickstart on Wed Sep  4 12:40:05 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Deep Neural Networks for Neurophysiology (DN3) Toolbox documentation!
====================================================================================

This Python package is an effort to bridge the gap between the neuroscience library MNE-Python and
the deep learning library PyTorch. This package's main focus is on minimizing boilerplate code, rapid deployment of
known solutions, and increasing the reproducibility of *new deep-learning solutions* for the analysis of M/EEG data
*(and may be compatible with other similar data... use at your own risk).*

Access to the code can be found at https://github.com/SPOClab-ca/dn3

Associated pre-print *(article under review)* can be found at:
 https://www.biorxiv.org/content/10.1101/2020.12.17.423197v1

*Please consider citing the above in any scholarly work that uses this library.*

.. image:: images/DN3-overview.*
   :alt: Overview of DN3 modules

The image above sketches out the structure of how the different modules of DN3 work together, but if you are new, we
recommend starting with the :doc:`configuration guide <../guides/configuration>`.

.. toctree::
   :glob:
   :caption: Guides
   :maxdepth: 1

   guides/*

.. toctree::
   :glob:
   :caption: Documentation
   :maxdepth: 2

   apidoc/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
