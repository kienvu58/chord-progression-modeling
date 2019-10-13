import torch
import torch.optim as optim
import numpy as np
import shutil
import itertools
import time
import math
import json
import logging
import sys
import os
import tinydb
import sys

from generate_similarity_target import generate_target

from modules.dataset_readers import CpmDatasetReader
from modules.chord_progression_models import Cpm
from modules.predictors import Predictor

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.training.trainer import Trainer
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
torch.manual_seed(1)


def get_default_hparams(fold=FOLD):
    hparams = {
        "name": NAME,
        "lr": 0.001,
        "batch_size": 32,
        "num_epochs": 200,
        "patience": 10,
        "rnn_type": "LSTM",
        "rnn_hidden_dim": 128,
        "fc_hidden_dim": 128,
        "chord_token_embedding_dim": 128,
        "training_mode": "one_hot",
        "similarity_matrix_path": None,
        "T_initial": 0.05,
        "decay_rate": 0.001,
        "data_path": "data/cv/{}/".format(fold),
        "vocab_path": "data/vocabulary/",
        "saved_model_path": "saved_models/",
    }
    return hparams

def update_hparams_for_grid_search_temperature_settings(T_initial, decay_rate, weight_set):
    hparams = get_default_hparams()
    hparams["training_mode"] = "decreased_temperature"
    hparams["similarity_matrix_path"] = get_similarity_matrix_path(weight_set)
    hparams["weight_set"] = weight_set
    hparams["T_initial"] = T_initial
    hparams["decay_rate"] = decay_rate
    return hparams

def update_hparams_for_fixed_temperature_training(T, weight_set):
    hparams = get_default_hparams()
    hparams["training_mode"] = "fixed_temperature"
    hparams["similarity_matrix_path"] = get_similarity_matrix_path(weight_set)
    hparams["weight_set"] = weight_set
    hparams["T_initial"] = T
    hparams["decay_rate"] = 0.0
    return hparams

def update_hparams_for_decreased_temperature_training(T_initial, decay_rate, weight_set):
    hparams = get_default_hparams()
    hparams["training_mode"] = "decreased_temperature"
    hparams["similarity_matrix_path"] = get_similarity_matrix_path(weight_set)
    hparams["weight_set"] = weight_set
    hparams["T_initial"] = T_initial
    hparams["decay_rate"] = decay_rate
    return hparams


def get_similarity_matrix_path(weight_set):
    path = "data/targets/target_{}.th".format(weight_set)
    if not os.path.exists(path):
        if "_" in weight_set:
            weight_set = weight_set.split("_")
            weight_set = [int(w) for w in weight_set]
            generate_target(weight_set)
    return path


def log(hparams, train_metrics, test_metrics, saved_model_path):
    if not os.path.isdir("logs/"):
        os.makedirs("logs/")
    
    db = tinydb.TinyDB("logs/logs.json")

    hparams["saved_model_path"] = saved_model_path
    hparams["train_perplexity"] = train_metrics["perplexity"]
    hparams["train_accuracy"] = train_metrics["accuracy"]
    hparams["train_loss"] = train_metrics["loss"]
    hparams["test_perplexity"] = test_metrics["perplexity"]
    hparams["test_accuracy"] = test_metrics["accuracy"]
    hparams["test_loss"] = test_metrics["loss"]
    hparams["timestamp"] = time.time()
    db.insert(hparams)


def get_contextualizer(rnn_type, input_dim, hidden_dim):

    if rnn_type == "RNN":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        )
    elif rnn_type == "LSTM":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        )
    elif rnn_type == "GRU":
        contextualizer = PytorchSeq2SeqWrapper(
            torch.nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        )
    else:
        raise ValueError("Unknown rnn type:", rnn_type)

    return contextualizer


def train_and_evaluate(hparams):
    # vocabulary
    vocab = Vocabulary().from_files(hparams["vocab_path"])
    vocab_size = vocab.get_vocab_size("tokens")

    # chord embedding
    token_embedding = Embedding(
        num_embeddings=vocab_size, embedding_dim=hparams["chord_token_embedding_dim"]
    )
    chord_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

    # data readers
    reader = CpmDatasetReader()
    train_dataset = reader.read(os.path.join(hparams["data_path"], "train.txt"))
    val_dataset = reader.read(os.path.join(hparams["data_path"], "val.txt"))
    test_dataset = reader.read(os.path.join(hparams["data_path"], "test.txt"))

    # contextualizer
    input_dim = chord_embedder.get_output_dim()
    hidden_dim = hparams["rnn_hidden_dim"]
    rnn_type = hparams["rnn_type"]
    contextualizer = get_contextualizer(rnn_type, input_dim, hidden_dim)

    # similarity matrix
    similarity_matrix_path = hparams["similarity_matrix_path"]
    if similarity_matrix_path is not None:
        similarity_matrix = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size,
            weight=torch.load(similarity_matrix_path),
            trainable=False,
        )
    else:
        similarity_matrix = None

    # training iterator
    batch_size = hparams["batch_size"]
    iterator = BucketIterator(
        batch_size=batch_size, sorting_keys=[("input_tokens", "num_tokens")]
    )
    iterator.index_with(vocab)
    batches_per_epoch = math.ceil(len(train_dataset) / batch_size)

    # model parameters
    model_hparams = {
        "dropout": None,
        "training_mode": hparams["training_mode"],
        "similarity_matrix": similarity_matrix,
        "T_initial": hparams["T_initial"],
        "decay_rate": hparams["decay_rate"],
        "batches_per_epoch": batches_per_epoch,
        "fc_hidden_dim": hparams["fc_hidden_dim"],
    }

    # chord progression model
    model = Cpm(vocab, chord_embedder, contextualizer, model_hparams)

    # check gpu available
    if torch.cuda.is_available():
        cuda_device = GPU_NO
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    # trainer
    ts = time.gmtime()
    saved_model_path = os.path.join(
        hparams["saved_model_path"], time.strftime("%Y-%m-%d %H-%M-%S", ts)
    )
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
    torch.save(model.state_dict(), os.path.join(saved_model_path, "best.th"))
    with open(os.path.join(saved_model_path, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=4)

    predictor = Predictor(model=model, iterator=iterator, cuda_device=cuda_device)
    train_metrics = predictor.predict(train_dataset)
    test_metrics = predictor.predict(test_dataset)
    log(hparams, train_metrics, test_metrics, saved_model_path)


def grid_search_temperature_settings(weight_set):
    global NAME
    NAME = "grid_search_temperature_for_{}".format(weight_set)
    T_initial_list = [0.01, 0.025, 0.05, 0.075, 0.1]
    decay_rate_list = [0.0005, 0.001, 0.0025, 0.005, 0.01]
    for T_initial, decay_rate in itertools.product(T_initial_list, decay_rate_list):
        hparams = update_hparams_for_grid_search_temperature_settings(T_initial, decay_rate, weight_set)
        train_and_evaluate(hparams)


def grid_search_weight_with_fixed_temperature(index, T_initial, decay_rate):
    weight_list = [32, 16, 8, 4, 2, 1]
    T_list = sample_temperature(10, 200, T_initial, decay_rate)
    for T, weight in itertools.product(T_list, weight_list):
        weight_set = [16, 0, 0, 0, 0, 0]
        weight_set[index] = weight
        weight_set = "_".join([str(w) for w in weight_set])
        hparams = update_hparams_for_fixed_temperature_training(T, weight_set)
        train_and_evaluate(hparams)


def run_with_fixed_temperature(weight_set, T_initial, decay_rate):
    T_list = sample_temperature(10, 200, T_initial, decay_rate)
    for T in itertools.product(T_list):
        hparams = update_hparams_for_fixed_temperature_training(T, weight_set)
        train_and_evaluate(hparams)


def grid_search_weight_with_decreased_temperature(T_initial, decay_rate):
    weight_list = [32, 16, 8, 4, 2, 1]
    index_list = [1, 2, 3, 4, 5]
    
    for weight, index in itertools.product(weight_list, index_list):
        weight_set = [16, 0, 0, 0, 0, 0]
        weight_set[index] = weight
        weight_set = "_".join([str(w) for w in weight_set])
        hparams = update_hparams_for_decreased_temperature_training(T_initial, decay_rate, weight_set)
        train_and_evaluate(hparams)


def sample_temperature(num_samples, num_epochs, T_intial, decay_rate):
    T = T_intial
    T_list = []
    for epoch in range(num_epochs):
        T *= 1 / (1 + decay_rate * epoch)
        T_list.append(T)

    step = num_epochs // num_samples
    return T_list[::step]
    

def train_with_one_hot():
    global NAME
    NAME = "train_with_one_hot"
    hparams = get_default_hparams()
    train_and_evaluate(hparams)


def train_with_magical_weights():
    global NAME
    NAME = "train_with_magical_weights"
    hparams = get_default_hparams()
    hparams["training_mode"] = "decreased_temperature"
    weight_set = "16_2_2_2_1_2"
    hparams["similarity_matrix_path"] = get_similarity_matrix_path(weight_set)
    hparams["weight_set"] = weight_set
    hparams["T_initial"] = 0.05
    hparams["decay_rate"] = 0.001
    train_and_evaluate(hparams)


    


if __name__ == "__main__":
    GPU_NO = 0
    NAME = "T_initial=0.05, decay_rate=0.001"
    FOLD = sys.argv[1]

    # grid_search_temperature_settings("1_1_1_1_1_1")
    T_initial = 0.05
    decay_rate = 0.001
    # grid_search_weight_with_fixed_temperature(1, T_initial, decay_rate)
    # grid_search_weight_with_fixed_temperature(2, T_initial, decay_rate)
    # grid_search_weight_with_fixed_temperature(3, T_initial, decay_rate)
    # grid_search_weight_with_fixed_temperature(4, T_initial, decay_rate)
    # grid_search_weight_with_fixed_temperature(5, T_initial, decay_rate)
    # grid_search_weight_with_decreased_temperature(T_initial, decay_rate)
    # train_with_one_hot()
    # train_with_magical_weights()
    run_with_fixed_temperature("1_1_1_1_1_1", T_initial, decay_rate)
    grid_search_temperature_settings("16_32_16_4_4_32")