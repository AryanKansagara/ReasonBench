# Research report: response-only output-head adaptation

## Scope and implementation

This separate experiment tests supervised instruction/format adaptation of the pretrained SmolLM2-135M-Instruct checkpoint, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. It does not train a VLM, apply reinforcement learning or update the full model.

The model backbone and original output projection are frozen. A rank-4 additive path (hidden -> 4 -> vocabulary) is attached to the output projection, adding 198,912 trainable parameters. Its second matrix is zero-initialized so initial logits match the frozen model. This is a custom low-rank output-head adapter, not PEFT library integration or attention-layer LoRA. Supervised next-token cross-entropy is computed only on assistant response tokens and the assistant terminator; prompt and padding labels are masked with -100. The underlying causal-language-model loss performs the token shift.

The data consists of integer addition/multiplication prompts with `<calc>a+b</calc>` or `<calc>a*b</calc>` responses. Seed 41 shuffles the finite tuple set into 48 training, 12 validation and 12 test examples. Exact tuples are disjoint, but vocabulary, operands and grammar overlap. This is a format/templated-task adaptation study, not evidence of broad numerical generalization. Test examples are not used in training or checkpoint selection.

The fixed recipe is 40 steps, batch size 4, AdamW learning rate 0.003, weight decay 0.01, and rank 4. Validation loss is measured every 10 steps; the best checkpoint is restored before final test inference. Both unadapted and adapted generation use the same test prompts, greedy decoding and a 32-token cap. The baseline test output is recorded before training, but the fixed recipe is not changed in response to it.

## Observed local run

Executed 2026-10-27 on macOS ARM64 CPU, PyTorch 2.10.0 and Transformers 4.57.1.

| Metric | Before adaptation | After adaptation |
| --- | ---: | ---: |
| Validation response-token loss | 0.92669 | 0.09348 |
| Strict numerical matches / 12 test examples | 0 | 11 |
| Invalid output formats / 12 | 12 | 1 |

Selected checkpoint: step 40. All 22 project unit tests passed; six specifically check adapter initialization, frozen-base gradients, response masking, padding and disjoint tuples. Raw baseline/adapted outputs and validation history are stored in `results/sft_metrics.json`.

## Interpretation and limitations

Most of the observed gain is consistent with learning a response protocol that the unadapted checkpoint did not follow under these prompts. This does not prove improved general reasoning. Twelve templated test examples, one seed and a short run cannot establish broad effectiveness. The original two-shot evaluation in `REPORT.md` uses different prompts/cases, so its 8/12 result must not be compared directly with this experiment's 11/12.

No preference optimization, RLHF, DPO, multimodal inputs, image generation, attention-layer adaptation, quantized fine-tuning or large-scale distributed training was performed. Next steps include unseen language templates, operand-range shifts, larger independent suites and repeated seeds. These are not implemented.

Source was developed with AI assistance; inspect the implementation and state your contribution accurately. A primary API reference is the [Hugging Face causal language modeling guide](https://huggingface.co/docs/transformers/main/en/tasks/language_modeling).
