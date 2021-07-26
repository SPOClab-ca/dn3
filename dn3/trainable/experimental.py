from .layers import *
from dn3.data.dataset import DN3ataset
from dn3.transforms.instance import InstanceTransform
from dn3.transforms.channels import map_named_channels_deep_1010, DEEP_1010_CHS_LISTING, SCALE_IND

from .processes import BaseProcess, StandardClassification
from .models import Classifier


class TVector(Classifier):

    def __init__(self, num_target_people=None, channels=len(DEEP_1010_CHS_LISTING), hidden_size=384, dropout=0.1,
                 ignored_inds=(SCALE_IND,), incoming_channels=None, norm_groups=16, return_tvectors=False):
        self.hidden_size = hidden_size
        self.num_target_people = num_target_people
        self.dropout = dropout
        super(TVector, self).__init__(num_target_people, None, channels, return_features=return_tvectors)
        self.ignored_ids = ignored_inds
        self.mapping = None if incoming_channels is None else map_named_channels_deep_1010(incoming_channels)

        def _make_td_layer(in_ch, out_ch, kernel, dilation):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation),
                nn.ReLU(),
                nn.GroupNorm(norm_groups, out_ch),
                nn.Dropout(dropout),
            )

        self.td_net = nn.Sequential(
            _make_td_layer(channels, hidden_size, 5, 1),
            _make_td_layer(hidden_size, hidden_size, 3, 2),
            _make_td_layer(hidden_size, hidden_size, 3, 3),
            _make_td_layer(hidden_size, hidden_size, 1, 1),
            _make_td_layer(hidden_size, hidden_size * 3, 1, 1),
        )

        # 3 * 2 -> td_net bottlenecks width at 3, 2 for mean and std pooling
        self.t_vector = self._make_ff_layer(hidden_size * 3 * 2, hidden_size)

    def _make_ff_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.LayerNorm(out_ch),
            nn.Dropout(self.dropout),
        )

    def make_new_classification_layer(self):
        if self.num_target_people is not None:
            self.classifier = nn.Sequential(
                self._make_ff_layer(self.hidden_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.num_target_people)
            )
        else:
            self.classifier = lambda x: x

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        self.freeze_features(unfreeze=not freeze_features)

    def save(self, filename, ignore_classifier=True):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        torch.save(state_dict, filename)

    @property
    def num_features_for_classification(self):
        return self.hidden_size

    def features_forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # Ignore e.g. scale values for the dataset to avoid easy identification
        if self.mapping is not None:
            x = (x.permute([0, 2, 1]) @ self.mapping).permute([0, 2, 1])
        x[:, self.ignored_ids, :] = 0
        for_pooling = self.td_net(x)
        pooled = torch.cat([for_pooling.mean(dim=-1), for_pooling.std(dim=-1)], dim=1)
        return self.t_vector(pooled)


class ClassificationWithTVectors(StandardClassification):

    def __init__(self, classifier: Classifier, tvector_model: TVector, loss_fn=None, cuda=False, metrics=None,
                 learning_rate=None,):
        super(ClassificationWithTVectors, self).__init__(classifier=classifier, tvector_model=tvector_model,
                                                         loss_fn=loss_fn, cuda=cuda, metrics=metrics,
                                                         learning_rate=learning_rate)

    def build_network(self, **kwargs):
        super(ClassificationWithTVectors, self).build_network(**kwargs)
        self.tvector_model.train(False)
        incoming = self.classifier.num_features_for_classification
        self.attn_tvect = torch.nn.Parameter(torch.ones((self.tvector_model.num_features_for_classification,
                                                         self.classifier.num_features_for_classification),
                                                        requires_grad=True, device=self.device))
        self.meta_classifier = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(incoming),
            # nn.Linear(incoming, incoming // 4),
            # nn.Dropout(0.6),
            # nn.Sigmoid(),
            nn.Linear(incoming, self.classifier.targets)
        )

    def train_step(self, *inputs):
        self.meta_classifier.train(True)
        return super(StandardClassification, self).train_step(*inputs)

    def evaluate(self, dataset, **loader_kwargs):
        self.tvector_model.train(False)
        self.meta_classifier.train(False)
        return super(ClassificationWithTVectors, self).evaluate(dataset)

    def parameters(self):
        yield from super(ClassificationWithTVectors, self).parameters()
        yield from self.meta_classifier.parameters()
        yield self.attn_tvect

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        _, classifier_features = self.classifier(inputs[0].clone())
        _, t_vectors = self.tvector_model(inputs[0])
        added = classifier_features.view(batch_size, -1) + t_vectors.view(batch_size, -1) @ self.attn_tvect
        return self.meta_classifier(added)


class TVectorConcatenation(InstanceTransform):

    def __init__(self, t_vector_model):
        if isinstance(t_vector_model, TVector):
            self.tvect = t_vector_model
        elif isinstance(t_vector_model, str):
            print("Loading T-Vectors model from path: {}".format(t_vector_model))
            self.tvect = TVector()
            self.tvect.load(t_vector_model)
        # ensure not in training mode
        self.tvect.train(False)
        for p in self.tvect.parameters():
            p.requires_grad = False
        super(TVectorConcatenation, self).__init__()

    def __call__(self, x):
        channels, sequence_length = x.shape
        tvector = self.tvect.features_forward(x).view(-1, 1)
        return torch.cat((tvector.expand(-1, sequence_length), x), dim=0)

    def new_channels(self, old_channels):
        return old_channels + ['T-vectors-{}'.format(i+1) for i in range(self.tvect.num_features_for_classification)]

