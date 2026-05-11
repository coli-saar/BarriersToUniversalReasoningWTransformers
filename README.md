# Barriers to Universal Reasoning With Transformers (And How to Overcome Them) 
This repository contains the code for the paper ["Barriers to Universal Reasoning With Transformers (And How to Overcome Them)"](https://arxiv.org/abs/2604.25800). 

### Repository Structure
* `src/lengthgen/tasks/`: Data generators for the evaluated tasks.
* `src/lengthgen/launch.py`: The entry point for running the from-scratch training of models + evaluation.
* `src/lengthgen/train.py`: From-scratch training file.
* `src/lengthgen/evaluate_model.py`: Script to test trained models across a range of sequence lengths.
* `src/lengthgen/prompting.py` & `launch_prompting.py`: Few-shot prompting evaluation scripts using either local Hugging Face models or the Together AI API.
* `src/lengthgen/build_tokenizer.py`: Script to generate custom tokenizers for the specific tasks.
* `src/lengthgen/scripts/`: Plotting and evaluation metric scripts.
* `submit`: Contains bash scripts to run training and evaluation
* `custom_tokenizers`: Contains white-space tokenizers specific for each task
* `config/`: Contains YAML configuration files for training runs, prompts for the prompting experiments, model configs, and evaluations.

### Environment Setup

You can install the required dependencies for the project by running `pip install -e .` inside the root directory. If you work on a GPU cluster, you can easily set up the environment by building a Docker image from the provided `Dockerfile`. The exact package versions are provided in `requirements.txt`.
Next, adjust the paths in `src/lengthgen/paths.py` and the `PROJECT_ROOT` and `SCRATCH_DIR` variables in the exe scripts under `submit/` to match your local file structure.

### Training Models
Training is managed via `src/lengthgen/launch.py`, which reads a YAML configuration file to run parameter sweeps.
You can launch a training run by using
```bash
./submit/exe_sweep.sh --config config/your_sweep_config.yaml
```
Example configs are provided under `config/tasks`.

*Note: Training uses Weights & Biases for tracking. Ensure you have set up a wandb API key and the other paths in exe_sweep.sh.*


### Evaluation
Once a model is trained, you can evaluate its accuracy on varying lengths using `evaluate_model.py`. 

```bash
./submit/exe_eval.sh
```
Make sure to adjust the parameters at the end of the file. 
This script then generates logs and CSVs outlining the model's accuracy per sequence length.

To run the full evaluation pipeline (i.e. selecting best model based on their val performance, running the evaluations, plotting the results), you can use 
```bash
./submit/exe_main_results.sh --config config/your_config.yaml
```
Example configs for this are provided under `config/results`

### Prompting Experiments
The prompting experiments in the paper are run via the Together AI API. This requires setting a Together API key.

```bash
export TOGETHER_API_KEY="your_api_key_here"
python src/lengthgen/launch_prompting.py --config config/your_prompt_config.yaml
```
Example configs are provided under `config/tasks/boolean_prompting.yaml` and `config/tasks/permutation_prompting.yaml`

You can also run the prompting experiments via local Huggingface models. To this end, exchange the `models` parameter in the yaml config, with `hf_models` and paste the Huggingface model string.


## Citation
If you find this project useful in your research, please consider citing our paper (to be updated after conference publication):
```bibtex
@article{kraus2026barriers,
  title={Barriers to Universal Reasoning With Transformers (And How to Overcome Them)},
  author={Kraus, Oliver and Sarrof, Yash and Yao, Yuekun and Koller, Alexander and Hahn, Michael},
  journal={arXiv preprint arXiv:2604.25800},
  year={2026}
}
```
