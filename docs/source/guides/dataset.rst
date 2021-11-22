.. _dataset_guide:

Datasets
=========================

.. contents:: :local:

For the most part, DN3 datasets are a simple wrapping around `MNE's <https://mne.tools/stable/python_reference.html>`_
`Epoch <https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs>`_ and
`Raw <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw>`_ objects, with the intent of:

  1. Providing a common API to minimize boilerplate around common divisions of data at the session, person and dataset
     boundaries
  2. Encouraging more consistency in data loading across multiple projects.
  3. Integration of (CPU bound) transformations operations, executed *on-the-fly* during deep network training

Thus there are three main interfaces:

  1. The :any:`Recording <_Recording>` classes

     - :any:`RawTorchRecording` and :any:`EpochTorchRecording`

  2. The :any:`Thinker` class, that collects a set of a *single person's* sessions
  3. The :any:`Dataset` class, that collects a set of multiple :any:`Thinker` that performed the same task under the
     same relevant context *(which in most cases means all the subjects of an experiment)*.

.. image:: ../images/data-layers.*
   :alt: Overview of data layers



Returning IDs
-------------

At many levels of abstraction, particularly for :any:`Thinker` and :any:`Dataset` , there is the option of returning
identifying values for the context of the trial within the larger dataset. In other words, the session, person, dataset,
and task ids can also be acquired while iterating through these datasets. These will always be returned sandwiched
between the actual recording value (first) and (if epoched) the class label for the recording (last), from *most
general* to *most specific*. Consider this example iteration over a :any:`Dataset`:

.. code-block:: python

   dataset = Dataset(thinkers, dataset_id=10, task_id=15, return_session_id=True, return_person_id=True,
                     return_dataset_id=True, return_task_id=True)

   for i, (x, task_id, ds_id, person_id, session_id, y) in enumerate(dataset):
       awesome_stuff()

Customized Loaders (e.g. dealing with .mat and .csv)
----------------------------------------------------

When trying to load auto-magically load a dataset with the configuratron (see the appropriate guide), it is not
a wholly uncommon occurance to find yourself dealing with a file-format exclusive to your own particular use-case.

For the most part, unless the world truly has no justice, the unique aspect of these
