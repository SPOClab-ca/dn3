#################
The Configuratron
#################
*High-level dataset and experiment descriptions*

.. contents:: :local:

Why do I need this?
===================
Configuration files are perhaps where the advantages of DN3 are most apparent. Ostensibly, integrating *multiple*
datasets to train a single process is as simple as loading files of each dataset from disk to be fed into a common
deep learning training loop. The reality however, is rarely that simple. DN3 uses `YAML <https://yaml.org/>`_ formatted
configuration files to streamline this process, and better organize the integration of *many* datasets.

Different file formats/extensions, sampling frequencies, directory structures make for annoying boilerplate with minor
variations. Here (among other possible uses) a consistent configuration framework helps to automatically handle
the variations across datatsets, for ease of integration down the road. If the dataset follows (or can be made to
follow) the relatively generic directory structure of session instances nested in a single directory for each unique
person, simply provided the top-level of this directory structure, a DN3 :any:`Dataset` can be rapidly constructed, with
easily adjustable *configuration* options.

Alternatively, if your dataset is all lumped into one folder, but follows a naming convention where the subject's name
and the session id are embedded in a consistent naming format, e.g. `My-Data-S01-R0.edf` and `My-Data-S02-R1.edf`, two
consistently formatted strings with two subjects (S01 and S02) and two runs (R0 and R1 - note that either subjects or
runs could also have been the same string and remained valid). In this case, you can use a (very *pythonic*) formatter
to organize the data hierarchically: `filename_format: "My-Data-{subject}-{session}"`

A Little More Specific
======================
Say we were evaluating a neural network architecture with some of our
own data. We are happy with how it is currently working, but want to now evaluate it against a public dataset to
compare with other work. Most of the time, this means writing a decent bit of code to load this new dataset. Instead,
DN3 proposes that it should be as simple as:

.. code-block:: yaml

   public_dataset:
     toplevel: /path/to/the/files

As far as the real *configuration* aspect, perhaps this dataset has a unique time window for its trials? Is the dataset
organized using filenames like the above "My-Data" example rather than directories? In that case:

.. code-block:: yaml

   public_dataset:
     toplevel: /path/to/the/files
     filename_format: "My-Data-{subject}-{session}"
     tmin: -0.1
     tlen: 1.5

Want to bandpass filter this data between 0.1Hz and 40Hz before use?

.. code-block:: yaml

   public_dataset:
     toplevel: /path/to/the/files
     filename_format: "My-Data-{subject}-{session}"
     tmin: -0.1
     tlen: 1.5
     hpf: 0.1
     lpf: 40


Hopefully this illustrates the advantage of organized datasets and configuration files, no boilerplate needed, you'll
get nicely prepared and consistent dataset abstractions (see :ref:`dataset_guide`). Not only this, but it allows for
people to share their *configurations*, for better reproducibility.

A Full Concrete Example
=======================
It takes a little more to make this a DN3 configuration, as we need to specify the existence of an experiment.
Don't panic, it's as simple as adding an empty **Configuratron** to the
yaml file that makes your configuration. Consider the contents of 'my_config.yml':

.. code-block:: yaml

   Configuratron:

   datasets:
     in_house_dataset:
       name: "Awesome data"
       tmin: -0.5
       tlen: 1.5
       picks:
         - eeg
         - emg

     public_dataset:
        toplevel: /path/to/the/files
        tmin: -0.1
        tlen: 1.5
        bandpass: [0.1, 40]

   architecture:
     layers: 2
     activation: 'relu'
     dropout: 0.1

The important entry here is `Configuratron`, that confirms this is an entry-point for the configuratron,
and `datasets` that lists the datasets we could use. The latter can either be named entries like the above,
or a list of unnamed entries.

Now, on the python side of things:

.. code-block:: python
   :emphasize-lines: 3,5

   from dn3.data.config import ExperimentConfig

   experiment = ExperimentConfig("my_config.yml")
   for ds_name, ds_config in experiment.datasets():
       dataset = ds_config.auto_construct_dataset()
       # Do some awesome things

The `dataset` variable above is now a DN3 :any:`Dataset`, which now readily supports loading trials for training or
separation according to people and/or sessions. Both the `in_house_dataset` and `public_dataset` will be available.

That's great, but what's up with that 'architecture' entry?
===========================================================
There isn't anything special to this, aside from providing a convenient location to add additional configuration
values that one might need for a set of experiments. These fields will now be populated in the `experiment` variable
above. So now, `experiment.architecture` is an object, with member variables populated from the yaml file.

MOAR! I'm a power user
======================
One of the really cool (my Mom says so) aspects of the configuratron is the addition of !include directives. Aside from
the top level of the file, you can include other files that can be readily reinterpreted as YAML, as supported by the
`pyyaml-include <https://github.com/tanbro/pyyaml-include>`_ project. This means one could specify all the available
datasets in one file called *datasets.yml* and include the complete listing for each configuration, say
*config_shallow.yml* and *config_deep.yml* by saying `datasets: !include datasets.yml`. Or you could include JSON
architecture configurations (potentially backed by your favourite cloud-based hyperparameter tracking module).

More directives might be added to the configuratron in the future, and we warmly welcome any suggestions/implementations
others may come up with.

Further, that `Configuratron` entry above also allows for a variety of experiment-level options, which allows for
common sets of channels, automatic adjustments of sampling frequencies and more. The trick is you need to keep reading.

Custom Loaders (e.g. handling .mat and .csv)
--------------------------------------------

The configuratron should cover the majority of use-cases, but as such, it probably doesn't cover every case. Adding a
function to a dataset config *before* using :any:`DatasetConfig.auto_construct_dataset()`, through:

  a. :any:`DatasetConfig.add_custom_raw_loader()` or :any:`DatasetConfig.add_custom_raw_loader()` to say,
     load a .mat file
  b. :any:`DatasetConfig.add_custom_thinker_loader()` to say, assemble a smattering of files into a single person's data

See the :any:`./dataset` guide for more information.


Complete listing of configuratron (experiment level) options
============================================================

Optional entries
----------------

use_only *(list)*
  A convenience option, whose purpose is to filter from datasets only the names in this list. This allows for inclusion
  of a large dataset file, and referencing certain named datasets. In this case, the names are the yaml key referencing
  the configuration.

deep1010 *(bool)*
  This will normalize and map all configuratron generated datasets using the :any:`MappingDeep1010` transform. This
  is on by default.

samples *(int)*
  Providing samples will enforce a global (common) length across all datasets (probably want to use this in conjunction
  with the *sfreq* option below).

sfreq *(float)*
  Enforce a global sampling frequency, down or upsampling loaded sessions if necessary. If a session cannot be
  downsampled without aliasing (it violates the nyquist criterion), a warning message will be printed, and the session
  will be skipped.

preload *(bool)*
  Whether to preload recordings for all datasets. *This is overridden by individual `preload` options
  for dataset configurations.

trial_ids *(bool)*
  Whether to return an id (*long tensor*) for which trial *within each recording* each data sequence returned by the
  constructed dataset.

relative_directory *(path)*
  This is an absolute path that, if provided, resolves any non-absolute paths used for `toplevel` paths for the
  datasets below.

Complete listing of dataset configuration fields
================================================

Required entries
----------------

toplevel *(required, directory)*
  Specifies the toplevel directory of the dataset. If a relative path, will be relative to the `relative_directory`
  of the `Configuratron` entry if provided, otherwise relative to working directory. Absolute paths will be used as is.

Special entries
---------------
**filename_format** *(str)*
  The special entry will assume that after scanning for all the correct *type* of file, the *subject* and *session*
  (or in DN3-speak, the *Thinker* and *Recording*) name can be parsed from the filepath. This should be a
  python-*format*-style string with two required substrings: *{subject}* and *{session}* that form a template for
  parsing subject and session ids from the path.
  Note, the file extension should not be included, and fixed length can
  be specified by trailing *:N* for length *N*, e.g. *{subject:2}* for specifically 2 characters devoted to subject ID.

The next few entries are superseded by the `Configuratron` entry *samples*, which defines a global number of samples
parameter. If this is not the case, **one of the following two is required**.

**tlen** *(required, float)*
  The length of time to use for each retrieved datapoint. If *epoched* trials (see :any:`EpochTorchRecording`) are
  required, *tmin* must also be specified.
**samples** *(required-ish, float)*
  As an alternative to tlen, for when you want to align datasets with pretty similar sampling frequencies, you can
  specify samples. If used, tlen is ignored (and not needed) and is inferred from the number of samples desired.

Optional entries
----------------

tmin *(float)*
  If specified, epochs the recordings into trials at each event (can be modified by *events* config below) onset with
  respect to *tmin*. So if *tmin* is negative, happens before the event marker, positive is after, and 0 is at the
  onset.
baseline *(list, None)*
  This option will only be used with epoched data (tmin is specified). This is simply propagated to the `Epoch's
  <https://mne.tools/stable/generated/mne.Epochs.html>`_ constructor as is. Where `None` can be specified using a tilde
  character: ~, as in *baseline: [~, ~]* to use all data for basline subtraction.
  **Unlike the default constructor, here by default, no baseline correction is performed.**
events *(list, map/dict)*
  This can be formatted in one of three ways:

  1. Unspecified - all events parsed by `find_events() <https://mne.tools/stable/generated/mne.find_events.html>`_,
     falling-back to `events_from_annotations() <https://mne.tools/stable/generated/mne.events_from_annotations.html>`_
  2. A list of event numbers that filter the set found from the above.
  3. A list of events (keys) and then labels (values) for those events, which filters as above, e.g.:

     .. code-block:: yaml

        events:
          T1: 5
          T2: 6

     The values should be integer codes, if both sides are numeric, this is used to map stim channel events to new
     values, otherwise (if the keys are strings), the annotations are searched.

  In all cases, the codes from the stim channel or annotations will not in fact correspond to the subsequent labels
  loaded. This is because the labels don't necessarily fit a minimal spanning set starting with 0. In other words, if
  I had say, 4 labels, they are not guaranteed to be 0, 1, 2 and 3 as is needed for loss functions downstream.

  The latter two configuration options above *do however* provide some control over this, with the order of the listed
  events corresponding to the index of the used label. e.g. *left_hand* and *right_hand* above have class labels
  0 and 1 respectively.

  If the reasoning for the above is not clear, not to worry. Just know you can't assume that annotated event 1 is label
  1. Instead use :meth:`EpochTorchRecording.get_mapping` to resolve labels to the original annotations or event codes.

force_label *(bool)*
  If set to `True`, will force the original epoch code (created using `events` above, or as determined by default
  values) to be returned by produced datasets. Otherwise, for N classes, will map them to labels 0 -> N-1.

annotation_format *(str)*
  In some cases, annotations may be provided as *separate* (commonly edf) files. This string should specify how to match
  the annotation file, optionally using the subject and session ids. This uses standard unix-style pattern matching,
  augmented with the ability to specify the subject with *{subject(:...)}* and the session with *{session(:...)}* as
  is used by filename_format. So one could use a pattern like: *"Data-*-{subject}-annotation"*. **Note, now by default,
  any file matching the annotation pattern is also excluded from being loaded as raw data.**

targets *(int)*
  The number of targets to classify if there are events. This is inferred otherwise.

chunk_duration *(float)*
  If specified, rather than using event offsets, create events every chunk_duration seconds, and then still use **tlen**
  and **tmin** with respect to these events. *This works with annotated recordings, and not recordings that rely on
  `stim` channels*.

picks *(list)*
  This option can take two forms:

   - The names of the desired channels
   - Channel types as used by `MNE's pick_types() <https://mne.tools/stable/generated/mne.pick_types.html>`_

  By default, will select only eeg and meg channels (if meg, will try to automatically resolve
  `as described here <https://mne.tools/stable/generated/mne.pick_types.html>`_)

exclude_channels *(list)*
  This is similar to the above, except it is a list of *nix pattern match exclusions. Which means it can be the channel
  names (that you want to exclude) themselves, or use wildards such as "FT*" or, "F[!39]". The first excludes all
  channels beginning with FT, the second, excludes all channels beginning with F *except* F3 and F9.

rename_channels *(dict)*
  Using this option, key's are the **new** name, and values are *nix-style pattern matching strings for the old channel
  names. *Warning* if an old channel matches to multiple new ones, new channel used is selected arbitrarily. Renaming
  is performed **before** exclusion.

decimate *(bool)*
  Only works with epoch data, must be > 0, default 1. Amount to decimate trials.

name *(string)*
  A more human-readable name for the dataset. This should be used to describe the dataset itself, not one of
  (potentially) many different configurations of said dataset (which might all share this parameter).

preload *(bool)*
  Whether to preload the recordings from this dataset. This overrides the experiment level `preload` option. Note that
  not all data formats support `preload`: False, but most do.

hpf *(float)*
  This entry (and the very similar `lpf` option) provide an option to highpass filter the raw data before anything else.
  It also supercedes any `preload`ing options, as the data needs to be loaded to perform this. It is specified in Hz.

lpf *(float)*
  This entry (and the very similar `hpf` option) provide an option to lowpass filter the raw data before anything else.
  It also supercedes any `preload`ing options, as the data needs to be loaded to perform this. It is specified in Hz.

extensions *(list)*
  The file extensions to seek out when searching for sessions in the dataset. These should include the '.', as in '.edf'
  . *This can include extensions not handled by auto_construction. A handler must then be provided using*
  :any:`DatasetConfig.add_extension_handler()`

stride *(int)*
  Only for :any:`RawTorchRecording`. The number of samples to slide forward for the next section of raw data. Defaults
  to 1, which means that each sample in the recording (aside from the last :samp:`sample_length - 1`) is used as the
  beginning of a retrieved section.

drop_bad *(bool)*
  Whether to ignore any events annotated as bad. Defaults to `False`

.. What am I doing about the filtering options?

data_max *(float, bool)*
  The maximum value taken by any recording in the dataset. Providing a float will assume this value, setting this to
  `True` instead automatically determines this value when loading data. These are required for a fully-specified use
  of the Deep1010 mapping.

  *CAUTION: this can be extremely slow. If specified, the value will be printed and should probably be explicitly added
  to the configuration subsequently.*

data_min *(float, bool)*
  The minimum value taken by any recording in the dataset. Providing a float will assume this value, setting this to
  `True` instead automatically determines this value when loading data. These are required for a fully-specified use
  of the Deep1010 mapping.

  *CAUTION: this can be extremely slow. If specified, the value will be printed and should probably be explicitly added
  to the configuration subsequently.*

dataset_id *(int)*
  This allows datasets to be given specific ids. By default, none are provided. If set to an int, this dataset will have
  this integer `dataset_id.

exclude_people *(list)*
  List of people (identified by the name of their respective directories) to be ignored. Supports Unix-style pattern
  matching *within quotations* (*, ?, [seq], [!seq]).

exclude_sessions *(list)*
  List of sessions (files) to be ignored when performing automatic constructions. Supports Unix-style pattern
  matching *within quotations* (*, ?, [seq], [!seq]).

exclude *(map/dict)*
  This is a more extensively formatted version of `exclude_people` and `exclude_sessions` from above. Here, people,
  sessions and timespans (specified in seconds) can be excluded using a hierarchical representation. The easiest way to
  understand this is by example. Consider:

  .. code-block:: yaml

        exclude_people:
          - Person01
        exclude:
          Person02: ~
          Person03:
            Session01: ~
          Person04:
            Session01:
              - [0, 0.5]
            Session02:
              - [0, 0.5]
              - [100, 120]

  The above says that *Person01* and *Person02* should both be completely ignored. *Session01* from *Person03* should be
  similarly ignored (with any other *Person03* session left available). Finally for *Person04* the data between
  0 and 0.5 seconds of *Session01* in addition to both the times between 0 and 0.5 and 100 and 120 seconds from
  *Session02* should be ignored. If

  In summary, it allows more fine-grained exclusion **without pattern matching**, and can be used in conjunction with
  the other exclusion options. For those familiar with MNE's *bads* system, it is not used here, this allows for config
  files to be shared rather than annotated copies of the original data. Further, this allows for easier by-hand editing.

Experimental/Risky Options
--------------------------

load_onthefly *(bool)*
  This overrides any preload values (for the dataset or experiment) and minimizes memory overhead from recordings
  at the cost of compute time and increased disk I/O. This is only really helpful if you have a dataset *so large* that
  mne's Raw instances start to fill your memory (this is not impossible, so if you are running out of memory, try
  switching on this option). Currently this does not work with epoched data.

moabb *(str)*
  This options allows you to specify the name of a `MOABB <https://github.com/NeuroTechX/moabb>`_ dataset. This is the
  one option for which `toplevel` is not in fact needed, as it will use the moabb-standard `~/mne_data` folder. If the
  dataset is missing, this should download the missing dataset.

  Specifying the toplevel does still work.

pre-dumped *(path)*
  Path to a directory where an optionally already preprocessed and/or transformed dataset has been saved. This is
  listed as a *risky* option, insofar as it ignores pretty much all of the rest of the configuration.

  See :any:`DN3ataset.to_numpy()` for how to dump the dataset to such a directory.