# ReasonBench

A bounded LLM inference/evaluation project comparing direct integer answers with a single calculator-tool call. Runs the real pretrained `HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint locally through Hugging Face Transformers and PyTorch.

## Reproduce

Python 3.12; CPU and about 1 GB of free memory plus model download/cache space are recommended. First run downloads public model files; weights are not redistributed in this repository.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python reasonbench.py --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --output artifacts
.venv/bin/python reasonbench.py --protocol zero-shot --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --output artifacts/initial
```

The default two-shot protocol gives two fixed examples per mode. Greedy decoding is capped at 48 new tokens. The checkpoint revision is pinned in these commands; when omitted, the script resolves and records the current revision. Loading uses safetensors and disables remote custom code.

## Implemented

- Twelve fixed arithmetic/word-arithmetic cases and two answer modes.
- A single calculator call in `<calc>expression</calc>` form. The model must propose the expression: the harness never substitutes an oracle expression or repairs a wrong proposal.
- An AST allowlist for bounded integer constants and `+ - * /`; rational arithmetic handles intermediate fractions. Names, calls, indexing, exponentiation, multi-call outputs and oversized expressions are rejected. No Python `eval`, shell, network tool or arbitrary code execution is used.
- Gold answers stay outside generation prompts and calculator execution. Strict scoring distinguishes valid wrong answers from invalid formats. A verbose correct answer can therefore be marked invalid; this measures protocol compliance as well as arithmetic.
- Per-case raw responses, checkpoint revision, settings, latency and summary metrics. Sixteen offline tests cover parser behavior, resource limits, wrong proposals and scoring.

## Observed results

Initial system-instruction protocol: 0/12 valid outputs in each mode. After a two-shot prompt revision: direct 2/12 correct, calculator 8/12 correct, with no invalid output formats. Both runs are preserved under `results/`. These are **development results on the same tiny fixed suite**, not held-out or statistically representative evidence of general reasoning improvement. See [report](docs/REPORT.md).

No fine-tuning, RL, DeepSpeed, distributed training, long-horizon planning, production deployment or internal Huawei product integration is implemented. The 135M checkpoint is a compact language model, not evidence of operating large production LLMs. Calculator execution only validates syntax and computes the proposal; it does not establish semantic correctness. This is a constrained tool-use experiment, not a general autonomous agent or security-certified sandbox.

The code was developed with AI assistance; review it and describe your contribution accurately. The inactive CI template runs offline unit tests; the model download/inference experiment is explicit, not an automatic external action.

Model/source reference: [official SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), Apache-2.0 licensed model. Results are local and are not model-card benchmark results.
