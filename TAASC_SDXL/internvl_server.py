"""Local InternVL3.5-HF scoring service for internvl_client.client_para.

Run with /home/zjs/anaconda3/envs/internvl/bin/python internvl_server.py.
The spelling internvl_server.py follows the requested filename.
Weights load once; CPU residency between requests leaves the GPU for SDXL.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DISK = Path('/media/zjs/ff61b503-6b67-47d0-bf89-a669c875d2f2')
POLICY = '''Evaluate ONLY the visible image against the requested annotations.
Annotations describe desired content, not facts about the image. Do not assume
an annotation is satisfied merely because it was requested.
Return exactly one JSON object with keys entities, attributes, relations.
Each value must be an array of numeric scores in [0,1], in the exact input order,
one score per annotation. Empty annotation groups require empty arrays.
Entities: confidence that the named object is visibly present.
Attributes [entity, attribute]: confidence that this attribute belongs to that
specific entity; do not transfer a color or property from a different object.
Relations [subject, predicate, object]: confidence in the specified directed
relation between the correct objects. Missing objects imply low confidence.
Score scale: 0 absent/contradicted, 0.25 mostly wrong, 0.5 ambiguous/partial,
0.75 mostly correct but uncertain, 0.9 clearly correct, 1.0 unequivocally correct.
Do not output explanations, markdown, or reasoning. Output JSON only.'''


def validate_annotation(annotation):
    if not isinstance(annotation, dict):
        raise ValueError('annotation must be an object')
    for group, width in [('entities', 1), ('attributes', 2), ('relations', 3)]:
        items = annotation.get(group)
        if not isinstance(items, list):
            raise ValueError(f'annotation.{group} must be an array')
        for item in items:
            values = [item] if width == 1 else item
            if not isinstance(values, list) or len(values) != width or not all(
                    isinstance(v, str) and v.strip() for v in values):
                raise ValueError(f'Invalid {group} entry: {item!r}')
    return {k: annotation[k] for k in ('entities', 'attributes', 'relations')}


def parse_scores(raw, annotation):
    # Handle harmless surrounding fences/thinking, but never invent missing scores.
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.S)
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[match.start():])
        except ValueError:
            continue
        if not isinstance(obj, dict) or not all(k in obj for k in annotation):
            continue
        result = {}
        for group, items in annotation.items():
            scores = obj[group]
            if not isinstance(scores, list) or len(scores) != len(items):
                raise ValueError(f'{group} needs {len(items)} scores, got {scores!r}')
            if any(isinstance(s, bool) or not isinstance(s, (int, float)) or
                   not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
                raise ValueError(f'Invalid numeric scores: {scores!r}')
            result[group] = [float(s) for s in scores]
        return result
    raise ValueError('No valid score JSON object in model output')


class Scorer:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation='sdpa').eval().requires_grad_(False)
        self.calls = 0
        self.audit = Path(args.audit_log)
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        logging.info('Model loaded once on CPU; ready on %s:%s', args.host, args.port)

    @torch.inference_mode()
    def score(self, image_path, annotation):
        annotation = validate_annotation(annotation)
        path = Path(image_path).expanduser().resolve(strict=True)
        # if not path.is_relative_to(Path(self.args.image_root).resolve()):
        #     raise ValueError('image_path must be inside --image-root')
        with self.lock:
            start = time.time()
            record = {'time_unix': start, 'image_path': str(path),
                      'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'annotation': annotation, 'model': self.args.model,
                      'policy': POLICY, 'generations': []}
            inputs = outputs = None
            try:
                with Image.open(path) as original:
                    image = original.convert('RGB')
                self.model.to('cuda')
                for attempt in range(2):
                    # Plain statements avoid the model mirroring input tuples as nested scores.
                    targets = {
                        'entities': [f'Object "{v}" is visibly present.' for v in annotation['entities']],
                        'attributes': [f'The "{entity}" has attribute "{attribute}".'
                                       for entity, attribute in annotation['attributes']],
                        'relations': [f'The "{subject}" {predicate} the "{obj}".'
                                      for subject, predicate, obj in annotation['relations']]}
                    prompt = POLICY + '\nScore these statements in order:\n' + json.dumps(targets, ensure_ascii=False)
                    prompt += '\nREQUIRED OUTPUT SHAPE:\n' + '\n'.join(
                        f'{group}: a FLAT array of exactly {len(items)} numbers (one number per statement).'
                        for group, items in annotation.items())
                    prompt += '\nNever use nested arrays. Each attribute statement gets ONE scalar score, not a pair.'
                    if attempt:
                        prompt += '\nYour previous response was invalid. Use the exact array lengths requested.'
                    record.setdefault('prompts', []).append(prompt)
                    messages = [{'role': 'user', 'content': [
                        {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
                    text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = self.processor(text=text, images=image, return_tensors='pt').to(
                        device='cuda', dtype=torch.bfloat16)
                    outputs = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens,
                                                  do_sample=False, use_cache=True)
                    raw = self.processor.batch_decode(outputs[:, inputs['input_ids'].shape[1]:],
                                                      skip_special_tokens=True)[0]
                    record['generations'].append(raw)
                    inputs = outputs = None
                    try:
                        result = parse_scores(raw, annotation)
                        break
                    except ValueError:
                        if attempt == 1:
                            raise
                record['scores'] = result
                self.calls += 1
                return result
            except Exception as error:
                record['error'] = str(error)
                raise
            finally:
                inputs = outputs = None
                self.model.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()
                record['elapsed_seconds'] = round(time.time() - start, 3)
                with self.audit.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                logging.info('Scored %s in %.1fs: %s', path.name, record['elapsed_seconds'],
                             record.get('scores', record.get('error')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(DISK / 'weights/InternVL3_5-8B-HF'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--image-root', default=str(DISK / 'experiment_outputs/paper2/TAASC_SDXL'))
    parser.add_argument('--audit-log', default=str(DISK / 'experiment_outputs/paper2/TAASC_SDXL/CC-500/service/internvl_responses.jsonl'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    scorer = Scorer(args)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, obj):
            payload = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error': 'not found'})
            self.reply(200, {'status': 'ok', 'model': args.model, 'calls': scorer.calls,
                             'busy': scorer.lock.locked(), 'idle_device': 'cpu'})

        def do_POST(self):
            if self.path != '/chat':
                return self.reply(404, {'error': 'not found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                result = scorer.score(data['image_path'], data['annotation'])
            except (ValueError, KeyError, FileNotFoundError) as error:
                return self.reply(400, {'error': str(error)})
            except Exception as error:
                logging.error('%s', traceback.format_exc())
                return self.reply(500, {'error': str(error)})
            self.reply(200, result)

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
