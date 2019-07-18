import torch
import os
from modules.data_preprocessors import parse_chord_name_core
from typing import Iterable
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from tqdm import tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric
import numpy as np
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator, cuda_device: int = -1):
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

        self.type_list = [
            "",
            "m",
            "o",
            "+",
            "7",
            "m7",
            "M7",
            "o7",
            "%7",
            "+7",
            "Ger6",
            "It6",
            "Fr6",
            "@end@",
        ]
        self.key_list = [
            "A",
            "Ab",
            "A#",
            "B",
            "Bb",
            "B#",
            "C",
            "Cb",
            "C#",
            "D",
            "Db",
            "D#",
            "E",
            "Eb",
            "E#",
            "F",
            "Fb",
            "F#",
            "G",
            "Gb",
            "G#",
            "@end@",
        ]
        self.init_confusion_matrices()

    def init_confusion_matrices(self):
        vocab_size = self.model.vocab.get_vocab_size()
        self.chord_cm = torch.zeros((vocab_size, vocab_size))
        self.type_cm = torch.zeros((len(self.type_list), len(self.type_list)))
        self.key_cm = torch.zeros((len(self.key_list), len(self.key_list)))

    def update_confusion_matrices(self, predictions, gold_labels):
        mask = gold_labels > 0
        predictions, gold_labels, mask = Metric.unwrap_to_tensors(
            predictions, gold_labels, mask
        )
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.size() - 1 but "
                "found tensor of shape: {}".format(predictions.size())
            )
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )
        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()

        top_k = predictions.max(-1)[1].unsqueeze(-1)
        gold_labels = gold_labels.unsqueeze(-1)

        vocab = self.model.vocab

        for i, gold_label in enumerate(gold_labels):
            if gold_label == 0:
                continue
            pred = top_k[i]
            gold_label, pred = gold_label.item(), pred.item()
            self.chord_cm[gold_label][pred] += 1

            gold_token = vocab.get_token_from_index(gold_label)
            pred_token = vocab.get_token_from_index(pred)

            gold_key, gold_form, gold_figbass = parse_chord_name_core(gold_token)
            pred_key, pred_form, pred_figbass = parse_chord_name_core(pred_token)

            if gold_key is None and gold_token == "@end@":
                gold_key = "@end@"
            if pred_key is None and pred_token == "@end@":
                pred_key = "@end@"

            if gold_key in self.key_list and pred_key in self.key_list:
                gold_key_idx = self.key_list.index(gold_key)
                pred_key_idx = self.key_list.index(pred_key)
                self.key_cm[gold_key_idx][pred_key_idx] += 1
            else:
                print((gold_token, gold_key), (pred_token, pred_key))

            if gold_key != "@end@":
                form = gold_form if gold_form is not None else ""
                figbass = gold_figbass if gold_figbass is not None else ""
                gold_type = form + figbass
            else:
                gold_type = "@end@"

            if pred_key != "@end@":
                form = pred_form if pred_form is not None else ""
                figbass = pred_figbass if pred_figbass is not None else ""
                pred_type = form + figbass
            else:
                pred_type = "@end@"

            if gold_type in self.type_list and pred_type in self.type_list:
                gold_type_idx = self.type_list.index(gold_type)
                pred_type_idx = self.type_list.index(pred_type)
                self.type_cm[gold_type_idx][pred_type_idx] += 1
            else:
                print((gold_token, gold_type), (pred_token, pred_type))

    def save_confusion_matrices(self):
        save_confusion_matrix_figure(
            "type_confusion_matrix",
            self.type_cm.numpy(),
            self.type_list,
            self.type_list,
        )
        save_confusion_matrix_figure(
            "key_confusion_matrix", self.key_cm.numpy(), self.key_list, self.key_list
        )

    def predict(self, dataset: Iterable[Instance]) -> None:
        self.init_confusion_matrices()
        pred_generator = self.iterator(dataset, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(
            pred_generator, total=self.iterator.get_num_batches(dataset)
        )

        self.model.eval()
        with torch.no_grad():
            batches_this_epoch = 0
            pred_loss = 0
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output_dict = self.model(**batch)

                predictions = output_dict["forward_logits"]
                gold_labels = batch.get("forward_output_tokens").get("tokens")
                self.update_confusion_matrices(predictions, gold_labels)

                loss = output_dict["loss"]
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    pred_loss += loss.detach().cpu().numpy()

                # Update the description with the latest metrics
                pred_metrics = training_util.get_metrics(
                    self.model, pred_loss, batches_this_epoch
                )
                description = training_util.description_from_metrics(pred_metrics)
                pred_generator_tqdm.set_description(description, refresh=False)

        return pred_metrics


def save_confusion_matrix_figure(name, matrix, xlabels, ylabels):
    plt.figure()

    plt.matshow(matrix, cmap="jet")
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=90, fontsize=5)
    plt.yticks(np.arange(len(ylabels)), ylabels, fontsize=5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, "{:d}".format(
                int(matrix[i][j])), ha="center", va="center", color="w", fontsize=5)

    plt.savefig(os.path.join("figures", "{}.pdf".format(name)))
