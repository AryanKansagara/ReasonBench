"""Bounded calculator-tool evaluation with real Hugging Face inference."""
import argparse
import ast
from fractions import Fraction
import json
from pathlib import Path
import re
import time

MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'


def calculate(expression):
    if not isinstance(expression, str) or not expression or len(expression) > 80:
        raise ValueError('invalid expression length')
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as exc:
        raise ValueError('invalid expression syntax') from exc
    if len(list(ast.walk(tree))) > 32:
        raise ValueError('expression too complex')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10000:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                if right == 0:
                    raise ValueError('division by zero')
                result = left / right
            if abs(result) > 1_000_000_000:
                raise ValueError('result out of range')
            return result
        raise ValueError('unsupported syntax')
    result = visit(tree.body)
    if result.denominator != 1:
        raise ValueError('expected integer result')
    return int(result)


def parse_direct(text):
    match = re.fullmatch(r'\s*(?:Answer:\s*)?(-?\d{1,10})(?:\.0)?[.!]?\s*', text)
    if not match:
        raise ValueError('invalid answer format')
    return int(match.group(1))


def parse_tool(text):
    match = re.fullmatch(r'\s*<calc>([^<>]{1,80})</calc>\s*', text)
    if not match:
        raise ValueError('invalid tool format')
    return calculate(match.group(1).strip())


def tasks():
    return [
        {'id': 'add', 'question': 'What is 17 + 26?', 'answer': 43},
        {'id': 'subtract', 'question': 'What is 91 - 38?', 'answer': 53},
        {'id': 'multiply', 'question': 'What is 13 * 7?', 'answer': 91},
        {'id': 'divide', 'question': 'What is 144 / 12?', 'answer': 12},
        {'id': 'precedence', 'question': 'What is 8 + 6 * 4?', 'answer': 32},
        {'id': 'parentheses', 'question': 'What is (19 + 5) * 3?', 'answer': 72},
        {'id': 'negative', 'question': 'What is 12 - 29?', 'answer': -17},
        {'id': 'combined', 'question': 'What is (84 - 12) / 6?', 'answer': 12},
        {'id': 'items', 'question': 'There are 7 boxes with 9 pens each. How many pens total?', 'answer': 63},
        {'id': 'remaining', 'question': 'A shop has 80 apples and sells 23. How many apples remain?', 'answer': 57},
        {'id': 'cost', 'question': 'Four tickets cost 15 dollars each. What is their total cost in dollars?', 'answer': 60},
        {'id': 'sharing', 'question': 'Sixty candies are shared equally among 5 children. How many candies per child?', 'answer': 12},
    ]


def prompt(question, mode, protocol='two-shot-v1'):
    if protocol == 'zero-shot':
        instruction = ('Answer with only one integer. Do not explain. Example: What is 2 + 3? Answer: 5.'
                       if mode == 'direct' else
                       'Return only a calculator call, exactly <calc>expression</calc>. '
                       'Use only integer numbers, +, -, *, /, and parentheses. '
                       'Example: What is 2 + 3? Output: <calc>2+3</calc>. Do not explain.')
        return [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': question}]
    if mode == 'direct':
        instruction = 'Answer with only one integer. Do not explain.'
        examples = [('What is 2 + 3?', '5'), ('Three bags contain 4 balls each. How many balls?', '12')]
    else:
        instruction = ('Return only a calculator call, exactly <calc>expression</calc>. '
                       'Use only integer numbers, +, -, *, /, and parentheses. '
                       'Do not explain.')
        examples = [('What is 2 + 3?', '<calc>2+3</calc>'),
                    ('Three bags contain 4 balls each. How many balls?', '<calc>3*4</calc>')]
    messages = [{'role': 'system', 'content': instruction}]
    for example, answer in examples:
        messages.extend([{'role': 'user', 'content': example + '\n' + instruction},
                         {'role': 'assistant', 'content': answer}])
    return messages + [{'role': 'user', 'content': question + '\n' + instruction}]


def assess(text, mode, answer):
    try:
        prediction = (parse_direct if mode == 'direct' else parse_tool)(text)
        return {'prediction': prediction, 'correct': prediction == answer,
                'status': 'correct' if prediction == answer else 'wrong_answer'}
    except ValueError as error:
        return {'prediction': None, 'correct': False, 'status': 'invalid_output', 'error': str(error)}


def summarize(records):
    result = {}
    for mode in ('direct', 'calculator'):
        group = [row for row in records if row['mode'] == mode]
        result[mode] = {'cases': len(group), 'correct': sum(row['correct'] for row in group),
                        'invalid_outputs': sum(row['status'] == 'invalid_output' for row in group),
                        'generation_seconds': sum(row['generation_seconds'] for row in group)}
    return result


def experiment(output, revision=None, protocol='two-shot-v1'):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.set_num_threads(4)
    revision = revision or HfApi().model_info(MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, trust_remote_code=False,
                                               use_safetensors=True).eval()
    records = []
    for mode in ('direct', 'calculator'):
        for task in tasks():
            # Gold answers are never passed to generation or tool execution.
            messages = prompt(task['question'], mode, protocol)
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_tensors='pt', return_dict=True)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=48, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            elapsed = time.perf_counter() - started
            text = tokenizer.decode(generated[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True)
            records.append({'id': task['id'], 'question': task['question'], 'answer': task['answer'],
                            'mode': mode, 'raw_output': text, 'generation_seconds': elapsed,
                            **assess(text, mode, task['answer'])})
    result = {'model': MODEL, 'revision': revision, 'protocol': protocol,
              'execution': 'CPU, greedy decoding, 48 new tokens maximum',
              'tool_budget': 1, 'summary': summarize(records), 'records': records,
              'limitations': ['12 fixed arithmetic cases, not a representative reasoning benchmark.',
                              'No fine-tuning, training, DeepSpeed, multi-step planning or external tool access.',
                              'Strict format parsing can mark a numerically correct verbose response invalid.',
                              'The calculator only checks syntax and computes the proposed expression; semantic correctness uses gold answers offline.']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'revision': revision, 'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--revision', default=None)
    parser.add_argument('--protocol', choices=['zero-shot', 'two-shot-v1'], default='two-shot-v1')
    args = parser.parse_args()
    experiment(args.output, args.revision, args.protocol)
