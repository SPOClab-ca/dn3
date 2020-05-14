Datasets
=========================

Datasets are how

.. contents:: :local:

Summary
--------
For the most part, DN3 datasets are a simple wrapping around `MNE's <https://mne.tools/stable/python_reference.html>`_
`Epoch <https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs>`_ and
`Raw <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw>`_ objects, with the intent of:

  1. Providing an interface for common divisions of data at the session, person and dataset boundaries
  2. Integration of (CPU bound) transformations operations, executed *on-the-fly* during deep network training

Thus there are three main interfaces:

  1. The :any:`_Recording` base class

     - With :any:`RawTorchRecording` and :any:`EpochTorchRecording` subclasses

  2. The :any:`Thinker` class, that collects a set of a *single person's* sessions
  3. The :any:`Dataset` class, that collects a set of multiple :any:`Thinker` that performed the same task under the
     same relevant context *(which in most cases means all the subjects of an experiment)*.
