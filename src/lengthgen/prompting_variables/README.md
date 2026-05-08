## Code for the Variable Assignment Experiments in the paper.

This folder generates simple "variable tracking" datasets, then we prompt a couple of models using the Together API.

### Packages required

```bash
pip install together tqdm
```

## Folder layout

- `datasets/len_{n_ops}/latent/dataset.jsonl` (raw underlying examples)
- `datasets/len_{n_ops}/final/direct/`
  - `none.jsonl`
  - `linenums.jsonl`
  - `linenums+value_change.jsonl`
- `results/len_{n_ops}/final/direct/{model_alias}/`
  - `{variant}{subset_tag}.json`
  - `{variant}{subset_tag}_summary.json`

### 1) Generate datasets (`data_python.py`)

From this folder:

```bash
python3 data_python.py \
  --output-root datasets \
  --n-examples 100 \
  --n-vars 3 \
  --n-ops 10 \
  --seed 0
```

What you get (example record fields): `prompt`, `gold_answer`, `program_lines`, plus metadata like `condition` and `variant_suffix`.

For `linenums.jsonl`, the `prompt`'s program is prefixed with line numbers like:

```text
Program:
 1. node_l = 8
 2. node_j = 9
 ...
```

### 2) Run Together eval (`run_api.py`)

Set your API key first:

```bash
export TOGETHER_API_KEY="YOUR_KEY_HERE"
```

Run a single configuration:

```bash
python3 run_api.py \
  --n-ops 10 \
  --variant linenums \
  --model llama70B
```

`--variant` choices are: `none`, `linenums`, `linenums+value_change`.

Optional: `--first-n`, `--sample-n`, `--start-idx`, `--end-idx`, and `--ids` let you evaluate only a subset.

### 3) Sweep configs (`run_models.sh`)

Run the default sweep over lengths `10 15 20 25 30` and variants:

```bash
./run_models.sh mistral24B
```

To change which model gets evaluated, pass the model alias as the first argument (e.g. `./run_models.sh llama70B`).

### Notes

- Paths are relative to the current working directory (so `cd prompting_variables` first).
