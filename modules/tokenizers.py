from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from .data_preprocessors import parse_chord_name_v2, convert_to_note_set


@Tokenizer.register("chord_character")
class ChordCharacterTokenizer(Tokenizer):
    """
    A ``ChordCharacterTokenizer`` splits strings into chord part tokens (key, form, figbass).
    """

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = [Token(t) for t in parse_chord_name_v2(text)]
        return tokens

@Tokenizer.register("note")
class NoteTokenizer(Tokenizer):
    """
    A ``NoteTokenizer`` splits strings into note tokens.
    """
    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = [Token(t) for t in convert_to_note_set(text)]
        return tokens