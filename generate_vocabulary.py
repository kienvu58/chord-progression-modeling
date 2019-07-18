import itertools

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN


def generate_vocab():
    note_list = ["A", "B", "C", "D", "E", "F", "G"]
    accidental_list = ["", "b", "#"]
    chord_type_list = ["", "m", "+", "o", "7", "m7", "M7", "o7", "%7", "+7", "It6", "Ger6", "Fr6"]

    vocab = Vocabulary()
    for ns in ["tokens", "token_in", "token_out"]:
        for chord in itertools.product(note_list, accidental_list, chord_type_list):
            vocab.add_token_to_namespace("".join(chord), namespace=ns)

        vocab.add_token_to_namespace(START_SYMBOL, namespace=ns)
        vocab.add_token_to_namespace(END_SYMBOL, namespace=ns)

    key_list = ["".join(x) for x in itertools.product(note_list, accidental_list)]
    form_list = ["m", "+", "o", "M", "%", "It", "Ger", "Fr"]
    figbass_list = ["7", "6"]
    for char in (key_list+form_list+figbass_list):
        vocab.add_token_to_namespace(char, namespace="token_characters")


    note_number_list = [str(x) for x in range(12)]
    for note_number in note_number_list:
        vocab.add_token_to_namespace(note_number, namespace="notes")

    vocab.save_to_files("data/vocabulary")

if __name__ == "__main__":
    generate_vocab()
    vocab = Vocabulary.from_files("data/vocabulary")
    print(vocab.get_token_to_index_vocabulary())

