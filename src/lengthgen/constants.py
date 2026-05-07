from enum import Enum

# Token that is used to mark the end of the prompt in a CoT trace. This token e.g. is used to mask out the loss of the prompt during training
TRACE_TOKEN = "<|trace|>"
# Token/sequence that is used to split raw text into prompt/reasoning trace for the prompting tasks
PROMPTING_TRACE_TOKEN = "### trace:"
# String/token used to denote the final answer in the reasoning trace
FINAL_ANSWER_TOKEN = "answer"
# String/token used to denote the final answer in the reasoning trace for the prompting tasks
FINAL_ANSWER_PROMPTING_TOKEN = "answer"
# Special token that is appended to every token in the vocab to mark a whitespace
WHITESPACE_TOKEN = "Ġ"
PADDING_TOKEN = "<|padding|>"
# end of text marker for the from-scratch tasks
END_OF_TEXT_TOKEN = "<|endoftext|>"
# end of text marker for the prompting tasks
END_OF_TEXT_PROMPTING_TOKEN = "end of result"

class PositionalEncodings(str, Enum):
    APE = 'ape'
    APE_SMALL = "ape_small"
    APE_XSMALL = "ape_xsmall"
