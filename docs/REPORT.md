# Research report: bounded calculator-assisted inference

## Research question

Can a compact pretrained instruction model reliably follow a calculator protocol, and does computing its proposed arithmetic expression differ from asking it to emit an integer directly? The task is deliberately small enough to inspect every output.

## Protocol and development history

Model: HuggingFaceTB/SmolLM2-135M-Instruct, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Local macOS ARM64 CPU, PyTorch 2.10.0, Transformers 4.57.1. Greedy decoding, at most 48 generated tokens, one calculator call maximum. No training or fine-tuning was performed.

The 12 fixed cases include arithmetic precedence, parentheses, negatives and short word problems. Direct outputs must parse as one integer. Tool outputs must contain exactly one allowed arithmetic expression wrapped in `<calc>` tags. The tool uses exact rational arithmetic and returns an integer. Gold values are used only by the scorer, never the generator or calculator. Examples in few-shot prompts are separate from the test cases but deliberately similar in task type.

The initial system-only format instruction failed: all outputs were verbose or otherwise invalid under both strict parsers. These results were retained in `results/initial_zero_shot.json`. We then revised prompts with two user/assistant examples and repeated the format instruction in the final user message. The revised protocol is `two-shot-v1`; its raw results are in `results/two_shot_v1.json`.

| Development protocol | Direct correct / 12 | Tool correct / 12 | Invalid direct / tool |
| --- | ---: | ---: | ---: |
| Initial system-only instruction | 0 | 0 | 12 / 12 |
| Two-shot-v1 | 2 | 8 | 0 / 0 |

The revised run spent about 1.17 seconds in direct generation and 2.60 seconds in tool-call generation across 12 cases each. This excludes model loading and is not a production latency study. Sixteen offline unit tests passed.

## Interpretation

The first failure was protocol compliance, not proof that every generated answer was numerically wrong: strict parsing rejects explanations. Few-shot conditioning changed both formatting and numerical behavior. The calculator path computes a model-proposed expression; it can still fail if the expression misrepresents the question. A successful computation is not a semantic verifier. A tiny prompt-sensitive model can also copy patterns from examples rather than robustly infer the task.

These are exploratory, in-sample prompt-development results. We inspected the same 12 cases between revisions; there is no untouched holdout set, confidence interval, sampling distribution or stronger-model baseline. The observed 8-versus-2 difference must not be presented as broad reasoning capability, a general agent improvement or a state-of-the-art result.

Next steps: freeze prompts and evaluate a separately generated, larger held-out suite; categorize expression-selection versus computation errors; compare stronger checkpoints and repeated seeds; add multi-step tasks only with explicit tool budgets and trace validation. None of these extensions is implemented.
