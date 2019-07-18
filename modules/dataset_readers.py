import logging
import os

from typing import Dict, List
from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField


class CpmDatasetReader(DatasetReader):
    """
    DatasetReader for Chord Progression Modelling data.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.tokenizer = tokenizer or WordTokenizer(
            word_splitter=JustSpacesWordSplitter()
        )
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        # No matter how you want to represent the input, we'll always represent the output as a
        # single token id.  This code lets you learn a language model that concatenates word
        # embeddings with character-level encoders, in order to predict the word token that comes
        # next.
        self.output_indexer: Dict[str, TokenIndexer] = None
        for name, indexer in self.token_indexers.items():
            if isinstance(indexer, SingleIdTokenIndexer):
                self.output_indexer = {name: indexer}
                break
        else:
            self.output_indexer = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as text_file:
            instance_strings = text_file.readlines()

        for sentence in instance_strings:
            yield self.text_to_instance(sentence)

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:
        sentence = "{} {} {}".format(START_SYMBOL, sentence, END_SYMBOL)
        tokenized_string = self.tokenizer.tokenize(sentence)
        input_field = TextField(tokenized_string[1:-1], self.token_indexers)
        forward_output_field = TextField(tokenized_string[2:], self.output_indexer)
        backward_output_field = TextField(tokenized_string[:-2], self.output_indexer)
        return Instance(
            {
                "input_tokens": input_field,
                "forward_output_tokens": forward_output_field,
                "backward_output_tokens": backward_output_field,
            }
        )
