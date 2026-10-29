"""Launch and record the full CC-500_deepseek experiment; --detach survives SSH exit."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
DISK = Path('/media/zjs/ff61b503-6b67-47d0-bf89-a669c875d2f2')
OUT = DISK / 'experiment_outputs/paper2/TAASC_SDXL/CC-500'
SERVICE = OUT / 'service'
SDXL = '/home/zjs/anaconda3/envs/sdxl/bin/python'
INTERNVL = '/home/zjs/anaconda3/envs/internvl/bin/python'
INPUT = DISK / 'datasets/paper2/CC-500_deepseek.jsonl'
COMMAND = [SDXL, '-u', str(ROOT / 'inference_sdxl.py'),
           '--model', str(DISK / 'weights/stable-diffusion-xl-base-1.0'),
           '--prompts_json', str(INPUT), '--outdir', str(OUT), '--save_attempts',
           '--disable_progress_bar']


def write_state(state):
    target = SERVICE / 'experiment_state.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    temp.replace(target)


def health():
    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
        return json.load(response)


def summarize():
    records = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]
    rows = []
    for sample in records:
        path = OUT / 'logs' / f'{sample["id"]}.json'
        report = json.loads(path.read_text()) if path.exists() else {}
        scores = report.get('final_scores') or {}
        values = [v for k in ('entities', 'attributes', 'relations') for v in scores.get(k, [])]
        rows.append({'id': sample['id'], 'caption': sample['caption'],
                     'complete': bool(report.get('complete') and (OUT / f'{sample["id"]}.png').is_file()),
                     'status': report.get('status', 'not_started'),
                     'rounds': len(report.get('rounds', [])),
                     'selected_round_1based': report.get('selected_round', -1) + 1,
                     'minimum_score': min(values) if values else '',
                     'mean_score': sum(values) / len(values) if values else ''})
    with (OUT / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'total': len(rows), 'completed': sum(r['complete'] for r in rows),
               'passed': sum(r['complete'] and r['status'] == 'passed' for r in rows),
               'max_rounds_reached': sum(r['complete'] and r['status'] == 'max_rounds_reached' for r in rows),
               'total_completed_rounds': sum(r['rounds'] for r in rows if r['complete'])}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def run():
    state = {'status': 'starting', 'supervisor_pid': os.getpid(), 'started_unix': time.time(),
             'command': COMMAND, 'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest()}
    write_state(state)
    server = None
    try:
        env = dict(os.environ, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   TOKENIZERS_PARALLELISM='false')
        snapshot = SERVICE / ('source_' + time.strftime('%Y%m%d_%H%M%S'))
        snapshot.mkdir()
        for file in ROOT.iterdir():
            if file.suffix in ('.py', '.sh', '.md'):
                shutil.copy2(file, snapshot / file.name)
        for name, python in [('sdxl', SDXL), ('internvl', INTERNVL)]:
            result = subprocess.run([python, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
            (snapshot / f'{name}_requirements.txt').write_text(result.stdout)
        try:
            info = health()
            if info.get('model') != str(DISK / 'weights/InternVL3_5-8B-HF'):
                raise RuntimeError('Port 8000 belongs to a different service')
        except OSError:
            server_log = (SERVICE / 'internvl_server.log').open('a')
            server = subprocess.Popen([INTERNVL, '-u', str(ROOT / 'internvl_server.py')],
                                      cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT)
            state['server_pid'] = server.pid
            write_state(state)
            for _ in range(180):
                if server.poll() is not None:
                    raise RuntimeError('InternVL server exited; inspect internvl_server.log')
                try:
                    info = health()
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise RuntimeError('InternVL server did not become ready in 360s')
        print('InternVL ready:', info, flush=True)
        with (SERVICE / 'inference.log').open('a') as log:
            process = subprocess.Popen(COMMAND, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', inference_pid=process.pid)
            write_state(state)
            code = process.wait()
        state.update(exit_code=code, summary=summarize())
        state['status'] = 'completed' if code == 0 and state['summary']['completed'] == state['summary']['total'] else 'failed'
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        state['finished_unix'] = time.time()
        write_state(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    SERVICE.mkdir(parents=True, exist_ok=True)
    if args.summary:
        print(json.dumps(summarize(), indent=2))
    elif args.detach:
        state_path = SERVICE / 'experiment_state.json'
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if previous.get('status') in ('starting', 'running'):
                try:
                    os.kill(previous['supervisor_pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise SystemExit('Experiment already running')
        with (SERVICE / 'supervisor.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                                      cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Started supervisor PID {process.pid}; logs: {SERVICE}')
    else:
        run()
