import json
import os

from lengthgen.constants import TRACE_TOKEN, FINAL_ANSWER_TOKEN, WHITESPACE_TOKEN, PADDING_TOKEN, END_OF_TEXT_TOKEN
from dataclasses import dataclass, field, asdict
from lengthgen.paths import TOKENIZER_BASE_PATH
from lengthgen.tasks import registry as task_registry
from transformers import PreTrainedTokenizer
from typing import List, Dict

TOKENIZER_OUT_BASE_PATH = str(TOKENIZER_BASE_PATH)
MAX_INDEX = 400

# maps the task type to its respetive tokenizer (e.g. useful when multiple tasks use the same tokenizer)
TASK_TO_TOKENIZER_DIR = {
    task_registry.TaskType.PERMUTATION: "permutation",
    task_registry.TaskType.PERMUTATION_BINARY: "permutation_binary",
    task_registry.TaskType.BOOLEAN: "boolean",
    task_registry.TaskType.PARITY: "parity",
}

@dataclass
class TaskVocabConfig:
    short_items: List[str] = field(default_factory=list)
    long_splits: Dict[str, List[str]] = field(default_factory=dict)
    box_labels: List[str] = field(default_factory=list)
    loss_weights: Dict[str, float] = field(default_factory=dict)
    structure_words: List[str] = field(default_factory=lambda: [
        "init", "operation", "swap", "write", "end", 
        "load", "line", "tape", "."
    ])

def _get_permutation_weights():
    weights = {
        f"{WHITESPACE_TOKEN}==": 7.0,
        f"{WHITESPACE_TOKEN}>": 7.0,
        f"{WHITESPACE_TOKEN}<": 7.0,
        f"{WHITESPACE_TOKEN}A": 7.0,
        f"{WHITESPACE_TOKEN}B": 7.0,
        f"{WHITESPACE_TOKEN}C": 7.0,
        f"{WHITESPACE_TOKEN}D": 7.0,
        f"{WHITESPACE_TOKEN}E": 7.0,
    }
    for i in range(MAX_INDEX):
        weights[f"{WHITESPACE_TOKEN}<{i}>"] = 2.0
    return weights



TASK_REGISTRY = {
    task_registry.TaskType.PERMUTATION: TaskVocabConfig(
        short_items=["Cat", "Dog", "Apple", "Book", "Hat", "A", "B", "C", "D", "E"
                     ],
        long_splits = {
            "Monkey":     [f"{WHITESPACE_TOKEN}Mon", "key"],
            "Dragon":     [f"{WHITESPACE_TOKEN}Dra", "gon"],
            "Spider":     [f"{WHITESPACE_TOKEN}Spi", "der"],
            "Armadillo":  [f"{WHITESPACE_TOKEN}Arma", "dillo"],
            "Salamander": [f"{WHITESPACE_TOKEN}Sala", "man", "der"],
            "Alligator":  [f"{WHITESPACE_TOKEN}All", "iga", "tor"],
            "Butterfly":  [f"{WHITESPACE_TOKEN}But", "ter", "fly"],
        },
        #loss_weights=_get_permutation_weights()
    ),
    task_registry.TaskType.PERMUTATION_BINARY: TaskVocabConfig(
        short_items=["Cat", "Dog", "Apple", "Book", "Hat", "A", "B", "C", "D", "E","K_A","K_B","K_C",
                     "K_D","K_E","W_A", "W_B","W_C","W_D","W_E","Cat_Dog","Dog_Cat", "==", ">", "<", "<A>","<B>","<C>","<D>","<E>", "IN","OUT",
                     "res","final"
                     ],
        loss_weights=_get_permutation_weights()
    ),
    task_registry.TaskType.BOOLEAN: TaskVocabConfig(
        short_items=["(",")","[","]",",","true","false","∧","∨","¬","T","F"],
    ),
    task_registry.TaskType.PARITY: TaskVocabConfig(
        short_items=["0", "1", "E", "O"],
    ),
}

class AlgorithmicTaskTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(
        self, 
        vocab_file=None,
        max_start_index=MAX_INDEX,
        long_splits=None, 
        short_items=None, 
        trace_token=TRACE_TOKEN,
        answer_token=FINAL_ANSWER_TOKEN,
        **kwargs
    ):
        self.max_start_index = max_start_index
        self.long_splits = long_splits or {}
        self.short_items = short_items or []
        
        self.trace_token = trace_token
        self.answer_token  = answer_token
        
        self.structure_words = kwargs.pop("structure_words", [])
        
        pad_tok = PADDING_TOKEN
        eos_tok = END_OF_TEXT_TOKEN
        unk_tok = "<|unk|>"
        
        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            self._vocab = self._build_vocab(pad_tok, eos_tok, unk_tok)
            
        self._id_to_token = {v: k for k, v in self._vocab.items()}

        kwargs["max_start_index"] = self.max_start_index
        kwargs["long_splits"] = self.long_splits
        kwargs["short_items"] = self.short_items
        kwargs["structure_words"] = self.structure_words
        kwargs["trace_token"] = self.trace_token

        kwargs.setdefault("pad_token", pad_tok)
        kwargs.setdefault("eos_token", eos_tok)
        kwargs.setdefault("unk_token", unk_tok)
        
        super().__init__(**kwargs)

    def _build_vocab(self,pad_tok,eos_tok,unk_tok):
        tokens = [pad_tok, eos_tok, unk_tok]

        tokens += [f"{WHITESPACE_TOKEN}{w}" for w in self.structure_words]
        tokens += [f"{WHITESPACE_TOKEN}{b}" for b in self.short_items]

        if self.trace_token and self.trace_token not in tokens:
            tokens.append(f"{WHITESPACE_TOKEN}{self.trace_token}")
        if self.answer_token and self.answer_token not in tokens:
            tokens.append(f"{WHITESPACE_TOKEN}{self.answer_token}")

        for n in range(0, self.max_start_index):
            tokens.append(f"{WHITESPACE_TOKEN}<{n}>")

        for parts in self.long_splits.values():
            tokens.extend(parts)

        seen = set()
        unique = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return {tok: idx for idx, tok in enumerate(unique)}

    def _tokenize(self, text: str, **kwargs):
        words = text.split()
        subtokens = []
        for word in words:
            if word in self.long_splits:
                subtokens.extend(self.long_splits[word])
            else:
                subtokens.append(f"{WHITESPACE_TOKEN}{word}")
        return subtokens

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens).replace(WHITESPACE_TOKEN, " ").lstrip()

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1 = None):
        return [0] * len(token_ids_0)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1 = None):
        # BOS/EOS already present in the generated text
        return token_ids_0

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        prefix = (filename_prefix + "-") if filename_prefix else ""
        vocab_file = os.path.join(save_directory, f"{prefix}vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        return (vocab_file,)

def main():
    for current_task, task_config in TASK_REGISTRY.items():
        print(f"Building tokenizer for: {current_task.value}")
        tok = AlgorithmicTaskTokenizer(**asdict(task_config))
        print(f"Vocab size: {tok.vocab_size}")
        
        if current_task == task_registry.TaskType.PERMUTATION:
            sample = (
                "init A Cat B Monkey C Dragon D Armadillo E Salamander "
                "operation line <3> swap A B . line <7> swap C D . end . "
                "load <3> . line <3> swap A B write A Monkey B Cat C Dragon D Armadillo E Salamander . "
                "end answer Monkey Cat Dragon Armadillo Salamander<|endoftext|>"
            )
            decoded = tok.decode(tok.encode(sample))
            print(f"Sample test match: {decoded.strip() == sample.strip()}")

        out_dir = f"{TOKENIZER_OUT_BASE_PATH}/{TASK_TO_TOKENIZER_DIR.get(current_task, current_task.value)}"
        tok.save_pretrained(out_dir)
        print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()

AlgorithmicTaskTokenizer.register_for_auto_class("AutoTokenizer")