from .layers import *
from dn3.data.dataset import DN3ataset
from dn3.transforms.channels import map_channels_deep_1010, DEEP_1010_CHS_LISTING, SCALE_IND

from .processes import BaseProcess, StandardClassification
from .models import DN3BaseModel


class DonchinSpeller(BaseProcess):

    def __init__(self, p300_detector: torch.nn.Module, detector_len: int, aggregator: torch.nn.Module, end_to_end=False,
                 loss_fn=None, cuda=False):
        self.detector = p300_detector
        self.detector_len = detector_len
        self.aggregator = aggregator
        self.loss = torch.nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        super().__init__(cuda=cuda)

    def build_network(self, **kwargs):
        pass

    def parameters(self):
        return *self.detector.parameters(), *self.aggregator.parameters()

    def train_step(self, *inputs):
        self.classifier.train(True)
        return super(BaseProcess, self).train_step(*inputs)

    def evaluate(self, dataset: DN3ataset):
        self.classifier.train(False)
        return super(BaseProcess, self).evaluate(dataset)

    def forward(self, *inputs):
        return self.classifier(inputs[0])

    def calculate_loss(self, inputs, outputs):
        return self.loss(outputs, inputs[-1])


class TVector(DN3BaseModel):

    def __init__(self, num_target_people, samples, channels=len(DEEP_1010_CHS_LISTING), hidden_size=384, dropout=0.1,
                 ignored_inds=(SCALE_IND,), incoming_channels=None):
        super().__init__(num_target_people, samples, channels)
        self.ignored_ids = ignored_inds
        self.mapping = None if incoming_channels is None else map_channels_deep_1010(incoming_channels)

        def _make_td_layer(in_ch, out_ch, kernel, dilation):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch),
                nn.Dropout(dropout),
            )

        self.td_net = nn.Sequential(
            _make_td_layer(channels, hidden_size, 5, 1),
            _make_td_layer(hidden_size, hidden_size, 3, 2),
            _make_td_layer(hidden_size, hidden_size, 3, 3),
            _make_td_layer(hidden_size, hidden_size, 1, 1),
            _make_td_layer(hidden_size, hidden_size * 3, 1, 1),
        )

        def _make_ff_layer(in_ch, out_ch):
            return nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch),
                nn.Dropout(dropout),
            )
        # 3 * 2 -> td_net bottlenecks width at 3, 2 for mean and std pooling
        self.t_vector = _make_ff_layer(hidden_size * 3 * 2, hidden_size)
        self.classifier = nn.Sequential(
            _make_ff_layer(hidden_size, hidden_size),
            nn.Linear(hidden_size, num_target_people)
        )

    @property
    def num_features_for_classification(self):
        return self.hidden_size

    def forward(self, x):
        # Ignore e.g. scale values for the dataset to avoid easy identification
        if self.mapping is not None:
            x = (x.permute([0, 2, 1]) @ self.mapping).permute([0, 2, 1])
        x[:, self.ignored_ids, :] = 0
        for_pooling = self.td_net(x)
        pooled = torch.cat([for_pooling.mean(dim=-1), for_pooling.std(dim=-1)], dim=1)
        t_vector = self.t_vector(pooled)
        return self.classifier(t_vector), t_vector


class ClassificationWithTVectors(StandardClassification):

    def __init__(self, classifier: DN3BaseModel, tvector_model: TVector, loss_fn=None, cuda=False, metrics=None,
                 learning_rate=None,):
        super(ClassificationWithTVectors, self).__init__(classifier=classifier, tvector_model=tvector_model,
                                                         loss_fn=loss_fn, cuda=cuda, metrics=metrics,
                                                         learning_rate=learning_rate)
        self.meta_classifier = nn.Sequential(
            Flatten(),
            nn.Linear(classifier.num_features_for_classification + tvector_model.num_features_for_classification,
                      classifier.targets)
        )

    def parameters(self):
        return super(ClassificationWithTVectors, self).parameters() + self.meta_classifier.parameters()

    def forward(self, *inputs):
        _, classifier_features = self.classifier(inputs[0])
        _, t_vectors = self.tvector_model(inputs[0])
        return self.meta_classifier(torch.stack((classifier_features, t_vectors), dim=-1))

