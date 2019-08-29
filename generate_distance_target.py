import torch
import numpy as np
import types
import os
import itertools
from allennlp.data.vocabulary import Vocabulary
from modules.data_preprocessors import (
    parse_chord_name_core,
    convert_to_note_set,
    convert_to_note_set_core,
    get_key_number,
)


def chord_to_pitch_class_vector(chord):
    vec = np.zeros((12,))
    pitch_classes = convert_to_note_set(chord)
    for note in pitch_classes:
        vec[int(note)] = 1

    return vec


def distance_0(chord_a, chord_b):
    if chord_a == chord_b:
        return 0
    return 1


def distance_1(chord_a, chord_b):
    return 0


def distance_2(chord_a, chord_b):
    vec_a = chord_to_pitch_class_vector(chord_a)
    vec_b = chord_to_pitch_class_vector(chord_b)
    dist = np.linalg.norm(vec_a - vec_b)
    return dist


def generate_distance_target(index, eps=1):
    vocab = Vocabulary().from_files("data/vocabulary")

    vocab_size = vocab.get_vocab_size()
    weight = np.zeros((vocab_size, vocab_size))

    if index == 0:
        dist_func = distance_0
    if index == 1:
        dist_func = distance_1
    if index == 2:
        dist_func = distance_2

    for i in range(vocab_size):
        chord_i = vocab.get_token_from_index(i)
        for j in range(vocab_size):
            chord_j = vocab.get_token_from_index(j)
            if "@" in chord_i or "@" in chord_j:
                M = 1 - distance_0(chord_i, chord_j)
            else:
                dist = dist_func(chord_i, chord_j)
                M = 1 / (dist + eps)
            weight[i][j] = M

    max_value = np.max(weight)
    weight /= max_value


    weight = torch.from_numpy(weight).float()
    if not os.path.isdir("data/targets/"):
        os.makedirs("data/targets/")

    torch.save(weight, "data/targets/target_distance_{}.th".format(index))


if __name__ == "__main__":
    generate_distance_target(2)