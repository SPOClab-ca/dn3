import torch
import numpy as np
from scipy.linalg import sqrtm
from dn3.transforms.instance import EuclideanAlignmentTransform, EEG_INDS


class Preprocessor:
    """
    Base class for various preprocessing actions. Sub-classes are called with a subclass of `_Recording`
    and operate on these instances in-place.

    Any modifications to data specifically should be implemented through a subclass of :any:`BaseTransform`, and
    returned by the method :meth:`get_transform()`
    """
    def __call__(self, recording, **kwargs):
        """
        Preprocess a particular recording. This is allowed to modify aspects of the recording in-place, but is not
        strictly advised.

        Parameters
        ----------
        recording :
        kwargs : dict
                 New :any:`_Recording` subclasses may need to provide additional arguments. This is here for support of
                 this.
        """
        raise NotImplementedError()

    def get_transform(self):
        """
        Generate and return any transform associated with this preprocessor. Should be used after applying this
        to a dataset, i.e. through :meth:`DN3ataset.preprocess`

        Returns
        -------
        transform : BaseTransform
        """
        raise NotImplementedError()


class EuclideanAlignmentPreprocessor(Preprocessor):
    """
    A session-wise implementation of He & Wu 2019; https://doi.org/10.1109/TBME.2019.2913914
    Used to some success with DNNs in Kostas & Rudzicz 2020; https://doi.org/10.1088/1741-2552/abb7a7

    Assumes that the dataset/session is already Deep1010 formatted with a mask (otherwise the empty channels ruin the
    calculation).
    """
    def __init__(self, inds=None, complex_tolerance=1e-4):
        self.reference_matrices = dict()
        self.ind_lookup = dict()
        self.fixed_inds = inds
        self._tol = complex_tolerance

    def __call__(self, session, session_id=0, thinker_id=0):
        if thinker_id in self.reference_matrices.keys():
            if session_id in self.reference_matrices[thinker_id].keys():
                raise ValueError(f"Already computed reference matrix for thinker {thinker_id}; session {session_id}")
        else:
            self.reference_matrices[thinker_id] = dict()
            self.ind_lookup[thinker_id] = dict()

        data = list()
        mask = list()
        # all_data = session.get_all()
        # data, mask = all_data[:2]

        for i in range(len(session)):
            x = session[i]
            mask.append(torch.cat([i for i in torch.nonzero(x[1]) if i in EEG_INDS]) if self.fixed_inds is None
                        else self.fixed_inds)
            data.append(x[0][mask[-1], :])

        # Masks have to be the same for the session for this to make any sense
        mask = mask[-1]
        data = torch.stack(data, dim=0)
        # data = data[:, mask, ...]
        #
        # inds = np.arange(session.channels.shape[0]) if self.fixed_inds is None else self.fixed_inds
        # data = torch.stack([session[i][0][inds, :] for i in range(len(session))], dim=0).double()
        data -= data.mean(axis=-1, keepdims=True)
        avg_cov = torch.mean(torch.matmul(data, torch.transpose(data, 2, 1)) / (data.shape[-1] - 1), dim=0).numpy()
        adjustment = sqrtm(avg_cov)

        if np.any(np.iscomplex(adjustment)):
            # Some wiggle room, needs some tolerance
            if np.max(np.imag(adjustment) / np.real(adjustment)) > self._tol:
                print("Warning: Sample covariance was not SPD somehow. Ignoring imaginary part.")
            adjustment = np.real(adjustment)

        # Pytorch currently doesn't have a well implemented matrix square root, so switch though scipy
        R = torch.inverse(torch.from_numpy(adjustment).float())
        self.reference_matrices[thinker_id][session_id] = R
        self.ind_lookup[thinker_id][session_id] = mask
        # return torch.matmul(R.T, data)

    def get_transform(self):
        if len(self.reference_matrices) == 0:
            raise ReferenceError('Preprocessor must be executed before the transform can be retrieved.')
        if len(self.reference_matrices) == 1:
            th = list(self.reference_matrices.keys())[0]
            if len(self.reference_matrices[th]) == 1:
                s = list(self.reference_matrices[th].keys())[0]
                return EuclideanAlignmentTransform(self.reference_matrices[th][s], self.ind_lookup[th][s])
        return EuclideanAlignmentTransform(self.reference_matrices, self.ind_lookup)
