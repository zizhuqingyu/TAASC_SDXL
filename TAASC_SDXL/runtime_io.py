"""Local input/model discovery and atomic results; no deep-learning imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from agent_utils import validate_sample, get_prompt_phrases


def resolve_model_path(value):
    path = Path(value).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in ('.safetensors', '.ckpt'):
            raise ValueError(f'Unsupported checkpoint: {path}')
        return path
    if (path / 'model_index.json').is_file():
        return path
    preferred = path / 'stable-diffusion-xl-base-1.0'
    if preferred.is_dir():
        return resolve_model_path(preferred)
    # Hugging Face cache: respect refs/main, instead of guessing by timestamp.
    roots = [path, path / 'models--stabilityai--stable-diffusion-xl-base-1.0']
    for root in roots:
        ref = root / 'refs' / 'main'
        if ref.is_file():
            revision = ref.read_text(encoding='utf-8').strip()
            if '/' in revision or '\\' in revision or revision in ('.', '..'):
                raise ValueError(f'Invalid cache revision in {ref}')
            snapshot = root / 'snapshots' / revision
            if (snapshot / 'model_index.json').is_file():
                return snapshot.resolve()
    candidates = set()
    for pattern in ('*/model_index.json', 'snapshots/*/model_index.json',
                    'models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/*/model_index.json'):
        candidates.update(p.parent.resolve() for p in path.glob(pattern))
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        raise ValueError(f'Multiple model directories found; specify --model exactly: {sorted(map(str,candidates))}')
    single = [p for p in path.glob('*.safetensors') if 'xl' in p.name.lower() and 'base' in p.name.lower()]
    if len(single) == 1:
        return single[0]
    raise FileNotFoundError(f'No complete SDXL pipeline found under {path}. '
                            'Expected model_index.json or stable-diffusion-xl-base-1.0/model_index.json; '
                            'a bare checkpoint also needs --single_file_config for offline loading.')


def resolve_variant(path, requested='auto'):
    if requested != 'auto':
        return requested or None
    if path.is_file():
        return None
    components = ('unet', 'vae', 'text_encoder', 'text_encoder_2')
    if all(list((path / name).glob('*.fp16.safetensors')) for name in components):
        return 'fp16'
    return None


def load_jsonl(path, allow_plain=False):
    samples, seen = [], set()
    with Path(path).expanduser().open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if allow_plain and isinstance(record, dict):
                    for group in ('entities', 'attributes', 'relations'):
                        record.setdefault(group, [])
                validate_sample(record)
                if not allow_plain and not get_prompt_phrases(record):
                    raise ValueError('Full TA-ASC requires nonempty structured semantic phrases')
                image_id = str(record['id'])
                if image_id in seen:
                    raise ValueError(f'Duplicate id: {image_id!r}')
                seen.add(image_id)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{path}, line {number}: {error}') from error
            samples.append(record)
    if not samples:
        raise ValueError('Input JSONL contains no records')
    return samples


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    image.save(temporary, format='PNG')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
