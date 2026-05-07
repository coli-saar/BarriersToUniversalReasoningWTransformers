import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()

# path where model configs are saved/read from
MODEL_CONFIG_BASE_PATH = Path(os.environ.get("MODEL_CONFIG_BASE_PATH", REPO_ROOT / "config/models"))
# path where tokenizers are saved to/loaded from
TOKENIZER_BASE_PATH = Path(os.environ.get("TOKENIZER_BASE_PATH", REPO_ROOT / "custom_tokenizers"))
# path where datasets are saved to
DATA_BASE_PATH = Path(os.environ.get("DATA_BASE_PATH", "/path/to/datasets"))
# path where trained models are saved to
MODELS_OUT_BASE_PATH = Path(os.environ.get("MODELS_BASE_PATH", "/path/to/models"))
# path where results are written to
RESULTS_OUT_BASE_PATH = Path(os.environ.get("RESULTS_BASE_PATH", REPO_ROOT / "results"))
# path to the train file
TRAIN_FILE_PATH = REPO_ROOT / "src" / "lengthgen" / "train.py"
# path to the eval file
EVAL_FILE_PATH = REPO_ROOT / "src" / "lengthgen" / "evaluate_model.py"
# path to the prompting entry script
PROMPTING_FILE_PATH = str(REPO_ROOT / "src" / "lengthgen" / "prompting.py")
# path to prompt templates
PROMPTS_BASE_PATH = Path(os.environ.get("PROMPTS_BASE_PATH", REPO_ROOT / "config" / "prompts"))
# either read HF cache from env variable, or set custom path
HF_CACHE_LOCATION = Path(os.environ.get("HF_CACHE_LOCATION", "/path/to/hf/cache"))
