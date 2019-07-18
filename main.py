import torch
import torch.optim as optim
import numpy as np
import shutil
import itertools
import json
import time
import math

from generate_vocabulary import generate_vocab
from generate_similarity_target import generate_target
from modules.tokenizers import ChordCharacterTokenizer, NoteTokenizer
from modules.dataset_readers import CpmDatasetReader
from modules.chord_progression_models import Cpm
from modules.predictors import Predictor

from allennlp.training.learning_rate_schedulers import CosineWithRestarts
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
)
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import (
    AugmentedLstm,
    BagOfEmbeddingsEncoder,
    CnnEncoder,
    CnnHighwayEncoder,
    PytorchSeq2VecWrapper,
)
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PytorchSeq2SeqWrapper,
    IntraSentenceAttentionEncoder,
    StackedSelfAttentionEncoder,
)
from allennlp.modules.similarity_functions import MultiHeadedSimilarity
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import (
    TokenIndexer,
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
)
from allennlp.training.trainer import Trainer
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

torch.manual_seed(1)


def run_experiment(use_similarity_targets, embedding_type, rnn_type, hparams):
    log = {}
    log["name"] = "{} {} {}".format(
        rnn_type, embedding_type, "similarity_target" if use_similarity_targets else "hard_target"
    )

    vocab = Vocabulary().from_files(hparams["vocab_path"])
    if embedding_type == "Chord":
        # data reader
        reader = CpmDatasetReader()

        # chord embedder
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size("tokens"),
            embedding_dim=hparams["chord_token_embedding_dim"],
        )
        chord_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

    elif embedding_type == "Note":
        # data reader
        note_tokenizer = NoteTokenizer()
        note_indexer = TokenCharactersIndexer(
            namespace="notes", min_padding_length=4, character_tokenizer=note_tokenizer
        )
        reader = CpmDatasetReader(
            token_indexers={"tokens": SingleIdTokenIndexer(),
                            "notes": note_indexer}
        )

        # chord embedder
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size("tokens"),
            embedding_dim=hparams["chord_token_embedding_dim"],
        )
        note_token_embedding = Embedding(
            vocab.get_vocab_size("notes"), hparams["note_embedding_dim"]
        )
        note_encoder = CnnEncoder(
            num_filters=hparams["cnn_encoder_num_filters"],
            ngram_filter_sizes=hparams["cnn_encoder_n_gram_filter_sizes"],
            embedding_dim=hparams["note_embedding_dim"],
            output_dim=hparams["note_level_embedding_dim"],
        )
        note_embedding = TokenCharactersEncoder(
            note_token_embedding, note_encoder)
        chord_embedder = BasicTextFieldEmbedder(
            {"tokens": token_embedding, "notes": note_embedding}
        )
    else:
        raise ValueError("Unknown embedding type:", embedding_type)

    # read data
    train_dataset = reader.read(os.path.join(
        hparams["data_path"], "train.txt"))
    val_dataset = reader.read(os.path.join(hparams["data_path"], "val.txt"))
    test_dataset = reader.read(os.path.join(hparams["data_path"], "test.txt"))

    # contextualizer
    contextual_input_dim = chord_embedder.get_output_dim()
    if rnn_type == "RNN":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.RNN(
                contextual_input_dim, hparams["rnn_hidden_dim"], batch_first=True, bidirectional=False
            )
        )
    elif rnn_type == "LSTM":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                contextual_input_dim, hparams["lstm_hidden_dim"], batch_first=True, bidirectional=False
            )
        )
    elif rnn_type == "GRU":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.GRU(
                contextual_input_dim, hparams["gru_hidden_dim"], batch_first=True, bidirectional=False
            )
        )
    else:
        raise ValueError("Unknown rnn type:", rnn_type)

    if use_similarity_targets:
        vocab_size = vocab.get_vocab_size("tokens")
        similarity_targets = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size,
            weight=torch.load(hparams["similarity_target_path"]),
            trainable=False,
        )
    else:
        similarity_targets = None

    iterator = BucketIterator(
        batch_size=hparams["batch_size"], sorting_keys=[
            ("input_tokens", "num_tokens")]
    )
    iterator.index_with(vocab)

    batches_per_epoch = math.ceil(len(train_dataset) / hparams["batch_size"])

    model_hparams = {
        "dropout": None,
        "similarity_targets": similarity_targets,
        "T_initial": hparams["T_initial"],
        "decay_rate": hparams["decay_rate"],
        "batches_per_epoch": batches_per_epoch,
        "fc_hidden_dim": hparams["fc_hidden_dim"]
    }
    # chord progression model
    model = Cpm(
        vocab,
        chord_embedder,
        contextualizer,
        model_hparams
    )

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
        print("GPU available.")
    else:
        cuda_device = -1

    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    ts = time.gmtime()
    saved_model_path = os.path.join(
        hparams["saved_model_path"], time.strftime("%Y-%m-%d %H-%M-%S", ts))
    serialization_dir = os.path.join(saved_model_path, "checkpoints")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        serialization_dir=serialization_dir,
        patience=hparams["patience"],
        num_epochs=hparams["num_epochs"],
        cuda_device=cuda_device,
    )
    trainer.train()
    saved_model_path = os.path.join(
        saved_model_path, "{}.th".format(log["name"]))
    torch.save(model.state_dict(), saved_model_path)

    predictor = Predictor(model=model, iterator=iterator,
                          cuda_device=cuda_device)
    pred_metrics = predictor.predict(test_dataset)
    log["metrics"] = pred_metrics
    log["saved_mode_path"] = saved_model_path

    return log


def main(hparams):
    if not os.path.isdir("logs/tmp/"):
        os.makedirs("logs/tmp/")
    similarity_target_path = hparams["similarity_target_path"]

    embedding_type_list = ["Chord"]
    rnn_type_list = ["LSTM", "GRU", "RNN"]
    use_similarity_targets = hparams["use_similarity_targets"]

    result = {}
    result["similarity_target"] = similarity_target_path if use_similarity_targets else None
    result["experiments"] = []
    for embedding_type, rnn_type in itertools.product(
        embedding_type_list, rnn_type_list
    ):
        log = run_experiment(use_similarity_targets,
                             embedding_type, rnn_type, hparams)
        result["experiments"].append(log)
        with open(os.path.join("logs", "tmp", "{} {}.json".format(log["name"], time.time())), "w") as f:
            json.dump(log, f, indent=4)

    result["hparams"] = hparams
    ts = time.gmtime()
    result_fn = "st={}, T0={}, lambda={}, time={}.json".format(
        similarity_target, T_initial, decay_rate, time.strftime("%Y-%m-%d %H-%M-%S", ts))

    with open(os.path.join("logs", result_fn), "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    if not os.path.isdir("data/vocabulary/"):
        generate_vocab()

    similarity_target_list = ["20-3-3-3-1-3",
                              "20-0-0-0-0-3", "20-3-3-3-1-0", None]
    for st in similarity_target_list:
        if st is None:
            continue
        if not os.path.exists("data/targets/target_{}.th".format(st)):
            weight_set = st.split("-")
            weight_set = [int(w) for w in weight_set]
            generate_target(weight_set)

    T_initial_list = [0.05]
    decay_rate_list = [0.001]
    data_fold_list = [0, 1, 2, 3, 4]

    for data_fold, similarity_target, T_initial, decay_rate in itertools.product(data_fold_list, similarity_target_list, T_initial_list, decay_rate_list):
        data_path = "data/cv/{}/".format(data_fold)
        vocab_path = "data/vocabulary/"
        saved_model_path = "saved_models/"
        similarity_target_path = "data/targets/target_{}.th".format(
            similarity_target)
        hparams = {
            "lr": 0.001,
            "batch_size": 32,
            "num_epochs": 200,
            "patience": 10,
            "rnn_hidden_dim": 128,
            "lstm_hidden_dim": 128,
            "gru_hidden_dim": 128,
            "fc_hidden_dim": 128,
            "chord_token_embedding_dim": 128,
            "note_embedding_dim": 64,
            "note_level_embedding_dim": 64,
            "cnn_encoder_num_filters": 16,
            "cnn_encoder_n_gram_filter_sizes": (2, 3, 4),
            "similarity_target_path": similarity_target_path,
            "T_initial": T_initial,
            "decay_rate": decay_rate,
            "data_path": data_path,
            "vocab_path": vocab_path,
            "saved_model_path": saved_model_path,
            "use_similarity_targets": True if similarity_target else False
        }
        main(hparams)
