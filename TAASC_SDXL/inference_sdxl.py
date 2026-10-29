"""TA-ASC on SDXL base: generate -> VLM score -> compensation -> regenerate.

运行：python inference_sdxl.py
权重默认读取 /home/zjs/dataset/weight 或其 stable-diffusion-xl-base-1.0 子目录。
JSON_PATH、OUTPUT_DIR 可在下方修改，也可使用 --prompts_json、--outdir 参数。

输入沿用 SD1.4 的结构化 JSONL：id、caption、entities、attributes、relations。
视觉评分沿用 client_para(image_path, annotation)；随附 HTTP 客户端默认访问
http://127.0.0.1:8000/chat。若已有不同协议的 internvl_client.py，使用原文件。
--disable_agent 仅生成一轮，不执行完整反馈闭环。

本版为 A6000 共卡运行修改：模型只读盘一次，按阶段搬移到 GPU。
默认第一轮 lambda=0，后续小幅补偿，并返回 VLM 评分最好的一轮。
这些是保守运行设置，不等同于原版论文参数及返回最后一轮的策略。
方法依据：1-Training-Free-260914.docx 第 3.1–3.4 节和给定 SD1.4 实现。
稳定性统计在上一步全层聚合完成后供下一步使用，N=3 最早第 5 步触发。
Run `python inference_sdxl.py --help` for all options.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path

from agent_utils import (build_annotation, build_agent_result, get_prompt_phrases,
                        load_agent_module, call_agent_with_retry, scores_all_reach_threshold,
                        update_lambda_value_dict, build_attention_feedback_targets)
from runtime_io import load_jsonl, resolve_model_path, resolve_variant, write_json, save_image, fingerprint

MODEL_PATH = '/media/zjs/ff61b503-6b67-47d0-bf89-a669c875d2f2/weights'
JSON_PATH = '/home/zjs/PycharmProjects/PythonProject/datasets/paper2/T2I-CompBench/complex_val_deepseek.jsonl'
OUTPUT_DIR = '/media/zjs/ff61b503-6b67-47d0-bf89-a669c875d2f2/experiment_outputs/paper2/T2I-CompBench/complex'


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', '--model_path', default=MODEL_PATH)
    p.add_argument('--prompts_json', '--json_path', default=JSON_PATH)
    p.add_argument('--outdir', default=OUTPUT_DIR)
    p.add_argument('--single_file_config', default=None, help='Local SDXL config/tokenizer directory for a bare checkpoint')
    p.add_argument('--variant', default='auto', help='auto, fp16, or empty string for normal filenames')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=('float16','bfloat16','float32'), default='float16')
    p.add_argument('--cpu_offload', action=argparse.BooleanOptionalAction, default=True,
                   help='Explicit phase offload: text -> U-Net -> VAE -> all CPU before VLM')
    p.add_argument('--vae_tiling', action='store_true', help='Optional VAE tiling for lower peak memory')
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--guidance_scale', type=float, default=7.0)
    p.add_argument('--guidance_rescale', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resample_each_attempt', action='store_true')
    p.add_argument('--negative_prompt', default=None, help='None matches the original sdxl_yuan.py')
    p.add_argument('--clip_skip', type=int, default=None)
    p.add_argument('--lambda_init', type=float, default=0.0)
    p.add_argument('--lambda_eta', type=float, default=0.1)
    p.add_argument('--lambda_max_step', type=float, default=0.02)
    p.add_argument('--lambda_max', type=float, default=0.25)
    p.add_argument('--score_threshold', type=float, default=0.8)
    p.add_argument('--max_agent_calls', type=int, default=10, help='Total generation/evaluation rounds, including the first')
    p.add_argument('--agent_module', default='internvl_client', help='Module exposing client_para(image_path, annotation)')
    p.add_argument('--agent_response_attempts', type=int, default=2)
    p.add_argument('--agent_url', default=None, help='Override INTERNVL_SERVER_URL for the supplied HTTP client')
    p.add_argument('--disable_agent', action='store_true', help='One generation only; not the full feedback method')
    p.add_argument('--attention_map_size', type=int, default=16)
    p.add_argument('--attention_stability_steps', type=int, choices=(3,4,5), default=3)
    p.add_argument('--attention_stability_threshold', type=float, default=0.85)
    p.add_argument('--attention_alpha', type=float, default=0.4)
    p.add_argument('--attention_beta', type=float, default=0.3)
    p.add_argument('--attention_gamma', type=float, default=0.3)
    p.add_argument('--attention_centroid_sigma', type=float, default=0.15)
    p.add_argument('--attention_high_response_ratio', type=float, default=0.2)
    p.add_argument('--attention_max_gain', type=float, default=0.15)
    p.add_argument('--mask_mode', choices=('paper','sd14'), default='paper')
    p.add_argument('--attention_query_chunk', type=int, default=256, help='0 = full query matrix')
    p.add_argument('--encode_batch_size', type=int, default=1)
    p.add_argument('--overlap_mode', choices=('mean','last'), default='mean')
    p.add_argument('--return_strategy', choices=('best','last'), default='best',
                   help='best: highest minimum phrase score, then mean; last: original paper policy')
    p.add_argument('--save_baseline', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--strict_alignment', action='store_true', help='Fail if any semantic phrase cannot be matched')
    p.add_argument('--save_attempts', action='store_true')
    p.add_argument('--skip_existing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--check_input', action='store_true', help='Validate JSONL without torch, model weights, or VLM')
    p.add_argument('--disable_progress_bar', action='store_true')
    return p


def validate_args(a):
    for name, value in vars(a).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'--{name} must be finite')
    for name in ('height','width'):
        if getattr(a,name) < 64 or getattr(a,name) % 8:
            raise ValueError(f'--{name} must be >= 64 and divisible by 8 (SDXL default: 1024)')
    for name in ('steps','max_agent_calls','agent_response_attempts','encode_batch_size'):
        if getattr(a,name) < 1:
            raise ValueError(f'--{name} must be positive')
    for name in ('lambda_init','lambda_max','attention_stability_threshold','guidance_rescale'):
        if not 0 <= getattr(a,name) <= 1:
            raise ValueError(f'--{name} must be in [0,1]')
    if a.lambda_init > a.lambda_max:
        raise ValueError('--lambda_init must not exceed --lambda_max')
    if not 0 < a.score_threshold <= 1 or not 0 < a.attention_high_response_ratio <= 1:
        raise ValueError('score_threshold and attention_high_response_ratio must be in (0,1]')
    for name in ('lambda_eta','lambda_max_step','attention_max_gain','attention_alpha',
                 'attention_beta','attention_gamma','attention_query_chunk','guidance_scale'):
        if getattr(a,name) < 0:
            raise ValueError(f'--{name} must be nonnegative')
    if a.attention_alpha + a.attention_beta + a.attention_gamma <= 0:
        raise ValueError('Stability coefficients must have a positive sum')
    if a.attention_centroid_sigma <= 0 or a.attention_map_size < 2:
        raise ValueError('Invalid stability sigma or attention map size')
    if a.clip_skip is not None and a.clip_skip < 0:
        raise ValueError('--clip_skip must be nonnegative')
    if a.limit is not None and a.limit < 1:
        raise ValueError('--limit must be positive')
    if not 0 <= a.seed < 2**63:
        raise ValueError('--seed must be in [0, 2**63)')


def load_pipeline(a):
    import torch
    from memory_runtime import PhasedSDXLPipeline, PhaseMemory
    device = torch.device(a.device)
    if device.type not in ('cuda','cpu'):
        raise ValueError('This runner supports CUDA or CPU')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; check the server PyTorch/driver installation')
    if device.type == 'cpu' and (a.dtype != 'float32' or a.cpu_offload):
        raise ValueError('CPU smoke test requires --dtype float32 --no-cpu_offload')
    path = resolve_model_path(a.model)
    options = dict(torch_dtype=getattr(torch,a.dtype), local_files_only=True,
                   use_safetensors=True, add_watermarker=False)
    variant = resolve_variant(path,a.variant)
    if path.is_file():
        if not a.single_file_config:
            raise ValueError('A single checkpoint needs --single_file_config with local SDXL configuration and tokenizers')
        pipe = PhasedSDXLPipeline.from_single_file(str(path), config=a.single_file_config, **options)
    else:
        if variant:
            options['variant'] = variant
        pipe = PhasedSDXLPipeline.from_pretrained(str(path), **options)
    for name in ('unet','vae','text_encoder','text_encoder_2'):
        model = getattr(pipe,name,None)
        if model is None:
            raise ValueError(f'Expected full SDXL base; missing {name}')
        model.eval().requires_grad_(False)
    if a.cpu_offload:
        # No Accelerate hooks: explicitly schedule the same model objects.
        for name in ('unet','vae','text_encoder','text_encoder_2'):
            getattr(pipe,name).to('cpu')
    else:
        pipe.to(device)
    pipe._taasc_memory = PhaseMemory(pipe,device,phased=a.cpu_offload)
    if a.vae_tiling:
        pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=a.disable_progress_bar)
    print(f'Model loaded ONCE: {path}; pipeline_object_id={id(pipe)}; '
          f'unet_object_id={id(pipe.unet)}; variant={variant}; scheduler={type(pipe.scheduler).__name__}', flush=True)
    pipe._taasc_memory.record('weights_loaded_once')
    return pipe, str(path)


def run_sample(pipe, controller, data, a, sample_seed, agent, output_dir, run_signature):
    import torch
    from taasc_embeddings import build_conditioning_cache
    image_id = str(data['id'])
    final_path = output_dir / f'{image_id}.png'
    report_path = output_dir / 'logs' / f'{image_id}.json'
    sample_signature = fingerprint({'sample':data, 'run':run_signature, 'sample_seed':sample_seed})
    if a.skip_existing and final_path.is_file() and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if previous.get('complete') and previous.get('signature') == sample_signature:
            print(f'Skip completed sample: {image_id}', flush=True)
            return
    phrases = get_prompt_phrases(data)
    weights = {phrase:a.lambda_init for phrase in phrases}
    memory = pipe._taasc_memory
    memory.history = []
    try:
        memory.enter('text_encoding', ('text_encoder','text_encoder_2'))
        cache = build_conditioning_cache(pipe, data['caption'], phrases, a.negative_prompt,
                         a.clip_skip, a.encode_batch_size, a.strict_alignment, a.overlap_mode)
    finally:
        memory.release('text_encoding_released')
    annotation = build_annotation(data)
    feedback = None
    max_rounds = 1 if a.disable_agent else a.max_agent_calls
    report = {'id':data['id'], 'caption':data['caption'], 'signature':sample_signature,
              'complete':False, 'status':'running', 'run':run_signature,
              'missing_phrases':cache.missing_phrases, 'token_positions':cache.position_info,
              'rounds':[], 'memory':memory.history, 'selection':a.return_strategy}
    best_rank = (-1.0,-1.0)
    best_image = None
    best_round = None
    write_json(report_path, report)
    for round_index in range(max_rounds):
        seed = (sample_seed + (round_index if a.resample_each_attempt else 0)) % (2**63)
        targets = build_attention_feedback_targets(data=data, agent_result=feedback,
                  position_info=cache.position_info, score_threshold=a.score_threshold,
                  max_gain=a.attention_max_gain)
        controller.configure(targets, batch_size=1, map_size=a.attention_map_size,
            stability_steps=a.attention_stability_steps, stability_threshold=a.attention_stability_threshold,
            alpha=a.attention_alpha, beta=a.attention_beta, gamma=a.attention_gamma,
            centroid_sigma=a.attention_centroid_sigma, high_response_ratio=a.attention_high_response_ratio,
            latent_height=a.height // pipe.vae_scale_factor, latent_width=a.width // pipe.vae_scale_factor,
            mask_mode=a.mask_mode)
        print(f'[{image_id}] round {round_index+1}/{max_rounds}, seed={seed}, lambda={weights}', flush=True)
        current_path = output_dir / 'attempts' / image_id / f'round_{round_index+1:02d}.png'
        # Recreate the same CPU generator each round to reproduce the same initial noise.
        # The official pipeline constructs SDXL pooled/time-size conditions and denoises.
        from memory_runtime import generate_image
        try:
            image = generate_image(pipe,controller,cache,weights,a,seed)
        except torch.cuda.OutOfMemoryError as error:
            report.update(status='generation_oom',error=str(error),complete=False)
            write_json(report_path,report)
            raise
        except Exception as error:
            report.update(status='generation_error',error=str(error),complete=False)
            write_json(report_path,report)
            raise
        save_image(image, current_path)
        if round_index == 0 and a.save_baseline:
            folder = 'baseline' if a.lambda_init == 0 else 'initial'
            save_image(image, output_dir / folder / f'{image_id}.png')
        detail = {'round':round_index, 'seed':seed, 'lambda':dict(weights), 'targets':targets,
                  'input_scores':feedback, 'output_scores':None, 'image':str(current_path),
                  'observed_layers':controller.observed_layers,
                  'enhanced_chunks':controller.enhanced_chunks, 'stability':controller.events}
        report['rounds'].append(detail)
        write_json(report_path,report)
        if a.disable_agent:
            report['status'] = 'single_pass_no_agent'
        else:
            try:
                memory.record('before_vlm_request')
                feedback = call_agent_with_retry(agent,current_path.resolve(),annotation,data,
                                                a.agent_response_attempts)
                memory.release_cache()
                memory.record('after_vlm_request')
            except Exception as error:
                report['status'] = 'agent_error'
                report['error'] = str(error)
                write_json(report_path,report)
                raise
            detail['output_scores'] = feedback
            from agent_utils import get_agent_score_group
            numeric = [float(value) for group in ('entities','attributes','relations')
                       for value in get_agent_score_group(feedback,group)]
            rank = (min(numeric),sum(numeric)/len(numeric))
            detail['selection_rank'] = list(rank)
            if rank > best_rank:
                best_rank, best_round = rank, round_index
                best_image = image.copy()  # PIL/CPU only; never keep a CUDA image.
            passed = scores_all_reach_threshold(feedback,a.score_threshold)
            report['status'] = 'passed' if passed else ('max_rounds_reached' if round_index+1 == max_rounds else 'running')
        write_json(report_path,report)
        if report['status'] != 'running':
            selected_round = best_round if a.return_strategy == 'best' and best_image is not None else round_index
            selected_image = best_image if a.return_strategy == 'best' and best_image is not None else image
            save_image(selected_image,final_path)
            selected = report['rounds'][selected_round]
            report.update(complete=True, final_image=str(final_path), final_lambda=selected['lambda'],
                          selected_round=selected_round, final_scores=selected['output_scores'])
            write_json(report_path,report)
            if not a.save_attempts:
                # Only exact files created by this completed run are removed.
                for entry in report['rounds']:
                    Path(entry['image']).unlink(missing_ok=True)
            print(f'Saved {final_path} ({report["status"]})', flush=True)
            del cache, best_image, image, selected_image
            controller.clear_tensor_state()
            memory.release('sample_finished')
            return
        weights = update_lambda_value_dict(build_agent_result(data,feedback),weights,
            target_score=a.score_threshold,eta=a.lambda_eta,max_step=a.lambda_max_step,lambda_max=a.lambda_max)


def main():
    a = build_parser().parse_args()
    validate_args(a)
    samples = load_jsonl(a.prompts_json,allow_plain=a.disable_agent)
    if a.limit:
        samples = samples[:a.limit]
    print(f'Validated {len(samples)} records from {a.prompts_json}', flush=True)
    if a.check_input:
        for sample in samples[:3]:
            print(json.dumps({'id':sample['id'],'phrases':get_prompt_phrases(sample)},ensure_ascii=False))
        return
    if a.agent_url:
        os.environ['INTERNVL_SERVER_URL'] = a.agent_url
    agent = None if a.disable_agent else load_agent_module(a.agent_module)
    pipe, model_path = load_pipeline(a)
    from taasc_attention import TrainingFreeAttentionController, install_taasc_attention_processors
    controller = TrainingFreeAttentionController()
    installation = install_taasc_attention_processors(pipe.unet,controller,a.attention_query_chunk)
    output_dir = Path(a.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    run_signature = {k:v for k,v in vars(a).items() if k not in
                     ('outdir','skip_existing','save_attempts','check_input','limit','disable_progress_bar','prompts_json')}
    run_signature['resolved_model'] = model_path
    run_signature['implementation'] = 'taasc-sdxl-a6000-20260917-v2'
    run_signature['effective_agent_url'] = os.getenv('INTERNVL_SERVER_URL','http://127.0.0.1:8000/chat')
    try:
        import diffusers, transformers, torch
        run_signature['versions'] = {'torch':torch.__version__, 'diffusers':diffusers.__version__,
                                     'transformers':transformers.__version__}
        for index, data in enumerate(samples):
            run_sample(pipe,controller,data,a,(a.seed+index) % (2**63),agent,output_dir,run_signature)
    finally:
        controller.clear_tensor_state()
        pipe._taasc_memory.release('runner_finished')
        installation.restore()


if __name__ == '__main__':
    main()
