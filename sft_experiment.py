"""Output-head-only supervised adaptation, not full-model SFT or RL."""
import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasonbench import MODEL, assess

REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'


class AdapterHead(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.base(hidden) + self.up(self.down(hidden))


def splits():
    rows = [{'a': a, 'b': b, 'op': op} for a in range(1, 13) for b in range(1, 13) for op in ('+', '*')]
    random.Random(41).shuffle(rows)
    return rows[:48], rows[48:60], rows[60:72]


def messages(row):
    return [{'role': 'user', 'content': f'What is {row["a"]} {row["op"]} {row["b"]}? Return only <calc>expression</calc>, without explanation.'}]


def mask_labels(ids, prefix):
    if not prefix or len(prefix) >= len(ids) or ids[:len(prefix)] != prefix:
        raise ValueError('invalid response token boundary')
    return [-100] * len(prefix) + ids[len(prefix):]


def encode(tokenizer, row):
    prompt = messages(row)
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    response = f'<calc>{row["a"]}{row["op"]}{row["b"]}</calc>'
    ids = tokenizer.apply_chat_template(prompt + [{'role': 'assistant', 'content': response}], tokenize=True)
    return ids, mask_labels(ids, prefix)


def batch(examples, pad):
    length = max(len(ids) for ids, _ in examples)
    return {'input_ids': torch.tensor([ids + [pad] * (length-len(ids)) for ids, _ in examples]),
            'attention_mask': torch.tensor([[1] * len(ids) + [0] * (length-len(ids)) for ids, _ in examples]),
            'labels': torch.tensor([labels + [-100] * (length-len(labels)) for ids, labels in examples])}


def response_loss(model, examples, pad):
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for index in range(0, len(examples), 4):
            part = examples[index:index+4]
            encoded = batch(part, pad)
            supervised = int((encoded['labels'][:, 1:] != -100).sum())
            total += model(**encoded, use_cache=False).loss.item() * supervised
            count += supervised
    return total / count


def generate_records(model, tokenizer, rows):
    model.eval()
    records = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(messages(row), tokenize=True, add_generation_prompt=True,
                                                return_tensors='pt', return_dict=True)
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True)
        expected = row['a'] + row['b'] if row['op'] == '+' else row['a'] * row['b']
        records.append({'input': row, 'raw_output': text, **assess(text, 'calculator', expected)})
    return {'correct': sum(r['correct'] for r in records), 'cases': len(rows),
            'invalid_outputs': sum(r['status'] == 'invalid_output' for r in records), 'records': records}


def experiment(output):
    torch.set_num_threads(4)
    torch.manual_seed(41)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False, use_safetensors=True)
    train, validation, test = splits()
    train_examples = [encode(tokenizer, row) for row in train]
    val_examples = [encode(tokenizer, row) for row in validation]
    baseline = generate_records(model, tokenizer, test)
    model.requires_grad_(False)
    head = AdapterHead(model.get_output_embeddings())
    model.set_output_embeddings(head)
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=0.003, weight_decay=0.01)
    initial_loss = response_loss(model, val_examples, tokenizer.eos_token_id)
    rng = random.Random(41)
    best, selected, state, history = float('inf'), 0, None, []
    for step in range(40):
        model.train()
        loss = model(**batch(rng.sample(train_examples, 4), tokenizer.eos_token_id), use_cache=False).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            val_loss = response_loss(model, val_examples, tokenizer.eos_token_id)
            history.append({'step': step + 1, 'validation_response_loss': val_loss})
            if val_loss < best:
                best, selected = val_loss, step + 1
                state = {'down.weight': head.down.weight.detach().clone(), 'up.weight': head.up.weight.detach().clone()}
    with torch.no_grad():
        head.down.weight.copy_(state['down.weight'])
        head.up.weight.copy_(state['up.weight'])
    adapted = generate_records(model, tokenizer, test)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(state, output / 'head_adapter.pt')
    result = {'model': MODEL, 'revision': REVISION, 'method': 'response-only SFT of rank-4 additive output-head adapter; frozen backbone',
              'train_examples': 48, 'validation_examples': 12, 'test_examples': 12,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'steps': 40, 'selected_step': selected, 'initial_validation_response_loss': initial_loss,
              'selected_validation_response_loss': best, 'history': history,
              'baseline': baseline, 'adapted': adapted,
              'limitations': ['Small templated arithmetic data; exact tuples disjoint but vocabulary and grammar shared.',
                              'No VLM, image inputs, full-model fine-tuning, attention-layer LoRA, RL or preference optimization.',
                              'Single seed, short run, tiny test set; not a general reasoning or adaptation benchmark.']}
    (output / 'sft_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('baseline','adapted','history')}, indent=2))
    print(json.dumps({name: {k: v for k, v in result[name].items() if k != 'records'} for name in ('baseline','adapted')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/sft'))
    experiment(parser.parse_args().output)
