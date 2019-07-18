import torch
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


class Score:
    def __init__(self, weight_set):
        self.score = {
            "token_name": weight_set[0],
            "key_name": weight_set[1],
            "key_number": weight_set[2],
            "triad_form": weight_set[3],
            "figbass": weight_set[4],
            "note_pair": weight_set[5],
        }

    def match_token_name(self, gold, pred):
        if gold == pred:
            return self.score["token_name"]
        return 0

    def match_key_name(self, gold, pred):
        gold_key, _, _ = parse_chord_name_core(gold)
        pred_key, _, _ = parse_chord_name_core(pred)
        if gold_key is not None and gold_key == pred_key:
            return self.score["key_name"]
        return 0

    def match_triad_form(self, gold, pred):
        _, gold_note_set = convert_to_note_set_core(gold)
        _, pred_note_set = convert_to_note_set_core(pred)
        if (
            gold_note_set is not None
            and pred_note_set is not None
            and gold_note_set[:3] == pred_note_set[:3]
        ):
            return self.score["triad_form"]
        return 0

    def match_figbass(self, gold, pred):
        gold_key, _, gold_figbass = parse_chord_name_core(gold)
        pred_key, _, pred_figbass = parse_chord_name_core(pred)
        if (
            gold_key is not None
            and pred_key is not None
            and gold_figbass == pred_figbass
        ):
            return self.score["figbass"]
        return 0

    def match_key_number(self, gold, pred):
        gold_key, _, _ = parse_chord_name_core(gold)
        pred_key, _, _ = parse_chord_name_core(pred)
        if (
            gold_key is not None
            and pred_key is not None
            and get_key_number(gold_key) == get_key_number(pred_key)
        ):
            return self.score["key_number"]
        return 0

    def match_note_pair(self, gold, pred):
        gold_note_set = convert_to_note_set(gold)
        pred_note_set = convert_to_note_set(pred)
        gold_note_pair_set = get_note_pair_set(gold_note_set)
        pred_note_pair_set = get_note_pair_set(pred_note_set)
        intersection = set(
            [note for note in gold_note_pair_set if note in pred_note_pair_set]
        )
        return len(intersection) * self.score["note_pair"]


def get_note_pair_set(note_set):
    pair_set = list(itertools.product(note_set, note_set))
    pair_set = set(
        [
            "_".join([str(note) for note in sorted(pair)])
            for pair in pair_set
            if pair[0] != pair[1]
        ]
    )
    return pair_set


def test_match_functions():
    score = Score([20, 3, 3, 3, 3, 3])
    assert score.match_token_name("AbM7", "AbM7") != 0
    assert score.match_token_name("@end@", "@end@") != 0
    assert score.match_key_name("FbM7", "Fbm") != 0
    assert score.match_key_name("FbM7", "FM7") == 0
    assert score.match_key_name("@end@", "FM7") == 0
    assert score.match_triad_form("Co7", "Fo") != 0
    assert score.match_triad_form("C7", "CM7") != 0
    assert score.match_triad_form("Cm7", "CM7") == 0
    assert score.match_figbass("CGer6", "FIt6") != 0
    assert score.match_figbass("F7", "Cm7") != 0
    assert score.match_figbass("F", "Cm") != 0
    assert score.match_figbass("@end@", "Cm") == 0
    assert score.match_key_number("G#", "Ab") != 0
    assert score.match_key_number("C", "B#") != 0
    assert score.match_key_number("B#", "B#") != 0
    assert score.match_key_number("G", "A") == 0
    assert score.match_note_pair("C", "C") == score.score["note_pair"] * 3
    assert score.match_note_pair("C", "Cm") == score.score["note_pair"]
    assert score.match_note_pair("C", "C7") == score.score["note_pair"] * 3
    assert score.match_note_pair("C7", "C7") == score.score["note_pair"] * 6


def get_target_distribution(score, gold_token, vocab):
    vocab_size = vocab.get_vocab_size()
    gold_index = vocab.get_token_index(gold_token)

    weight = torch.zeros((vocab_size,), dtype=torch.float)
    for index in range(vocab_size):
        token = vocab.get_token_from_index(index)
        match_func_list = [func for func in dir(Score) if func.startswith("match")]
        s = sum(
            [
                getattr(score, func_name)(gold_token, token)
                for func_name in match_func_list
            ]
        )
        weight[index] = s

    weight /= weight.sum()
    return weight


def generate_target(weight_set):
    vocab = Vocabulary().from_files("data/vocabulary")
    score = Score(weight_set)

    token_weight_list = []
    for index, token in vocab.get_index_to_token_vocabulary().items():
        token_weight = get_target_distribution(score, token, vocab)
        token_weight_list.append(token_weight)

    weight = torch.stack(token_weight_list)

    if not os.path.isdir("data/targets/"):
        os.makedirs("data/targets/")

    s = score.score
    torch.save(
        weight,
        "data/targets/target_{}-{}-{}-{}-{}-{}.th".format(
            s["token_name"],
            s["key_name"],
            s["key_number"],
            s["triad_form"],
            s["figbass"],
            s["note_pair"],
        ),
    )

if __name__ == "__main__":
    test_match_functions()

