Transformations and Preprocessors
=================================

.. contents:: :local:

Summary
-------
One of the advantages of using `PyTorch <https://pytorch.org/>`_ as the underlying computation library, is eager graph
execution that can leverage native python. In other words, it lets us integrate arbitrary operations in a largely
parallel fashion to our training (*particularly if we are using the GPU for any neural networks*).

Enter the :any:`BaseTransform` and its subclasses. When added to a :any:`DN3ataset`, these perform operations on each
fetched recording sequence, be it a trial or cropped sequence of raw data. For the most part, they are simply callable
objects, implementing :py:func:`__call__` to modify a :any:`Tensor` unless they modify the number/representation of
channels, sampling frequency or sequence length of the data.

:any:`Preprocessor` (s) are simply a method to create a transform after first encountering all of the :any:`_Recording`
(s) of a :any:`DN3ataset`. Simply put, if the transform is known *a priori*, the :any:`BaseTransform` interface is
sufficient. Otherwise, a :any:`Preprocessor` can be used to both modify :any:`_Recording` (s) in place *before*
training, and create a transformation to modify sequences *on-the-fly*.

