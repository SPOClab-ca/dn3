Transformations and Preprocessors
=================================

.. contents:: :local:

Summary
-------
One of the advantages of using `PyTorch <https://pytorch.org/>`_ as the underlying computation library, is eager graph
execution that can leverage native python. In other words, it lets us integrate arbitrary operations in a largely
parallel fashion to our training (*particularly if we are using the GPU for any neural networks*).

Instance Transforms
-------------------

Enter the :any:`InstanceTransform` and its subclasses. When added to a :any:`Dataset <DN3ataset>`, these perform
operations on each
fetched recording sequence, be it a trial or cropped sequence of raw data. For the most part, they are simply callable
objects, implementing :py:func:`__call__` to modify a :any:`Tensor` unless they modify the number/representation of
channels, sampling frequency or sequence length of the data.

They are specifically *instance* transforms, because they do not transform more than a single crop of data (from a
single person and dataset). This means, that these are done before a batch is aggregated for training. If the
transform results in many differently shaped tensors, **a batch will not properly be created, so watch out for that**!

Batch Transforms
----------------

These are the exceptions that prove the :any:`InstanceTransform` rule. These transforms operate only *after* data has
been aggregated into a batch, and it is just about to be fed into a network for training (or otherwise). These are
attached to trainable :any:`Processess <BaseProcess>` instead of :any:`Datasets <DN3ataset>`.

Multiple Worker Processes Warning
---------------------------------
After attaching enough transforms, you may find that, even with most of the deep learning side being done on the GPU
loading the training data may become the bottleneck.

Preprocessors
-------------
:any:`Preprocessor` (s) on the other hand are a method to *create* a transform after first encountering all of the
:any:`Recordings <_Recording>` of a :any:`Dataset <DN3ataset>`. Simply put, if the transform is known *a priori*, the
:any:`BaseTransform` interface is sufficient. Otherwise, a :any:`Preprocessor` can be used to both modify
:any:`Recordings <_Recording>` in place *before*
training, and create a transformation to modify sequences *on-the-fly*.

