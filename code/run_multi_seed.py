from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from statistics import mean, stdev
import numpy as np
import torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)
from eval_infer import discover_datasets
from main import METRICS, run_inference, run_training
from train import add_training_args
from utils.utils import format_metric, round_metric
torch.backends.cudnn.enabled = False
DEFAULT_SEEDS = (42, 123, 456, 789, 2024)
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def parse_args():
    parser = argparse.ArgumentParser(description='多种子实验：训练 + 测试 + 汇总')
    parser.add_argument('project', type=str, help='实验基础名称，各 seed 结果为 {project}_seed{seed}')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS), help=f'随机种子列表，默认 {list(DEFAULT_SEEDS)}')
    add_training_args(parser)
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    parser.add_argument('--skip_train', action='store_true', help='跳过训练，仅对已有 checkpoint 做测试与汇总')
    parser.add_argument('--output_root', type=str, default='./multi_seed_results', help='多种子汇总根目录')
    return parser.parse_args()
def project_for_seed(base_project: str, seed: int) -> str:
    return f'{base_project}_seed{seed}'
def read_summary_csv(path: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset'].strip()
            rows[dataset] = {k.strip(): round_metric(v) for k, v in row.items() if k and k.strip() != 'dataset'}
    return rows
def copy_summary(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
        fout.write(fin.read())
def run_one_seed(args, base_project: str, seed: int, out_dir: str) -> dict:
    proj = project_for_seed(base_project, seed)
    args.project = proj
    args.seed = seed
    seed_dir = os.path.join(out_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)
    set_global_seed(seed)
    print('\n' + '=' * 70)
    print(f'[Seed {seed}] project={proj}')
    print('=' * 70)
    best_path = os.path.join('snapshots', proj, 'best.pth')
    train_meta = {'seed': seed, 'project': proj}
    if args.skip_train:
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f'seed={seed} 未找到权重: {best_path}')
        train_meta['skipped_train'] = True
    else:
        train_result = run_training(args)
        best_path = train_result['best_path']
        train_meta['best_checkpoint'] = best_path
        train_meta['best_epoch'] = train_result['best_epoch']
        train_meta['best_val_sid'] = round_metric(train_result['best_val_sid'])
    val_split_path = os.path.join('snapshots', proj, 'val_split.json')
    if os.path.isfile(val_split_path):
        with open(val_split_path, encoding='utf-8') as f:
            split_info = json.load(f)
        train_meta['val_split'] = {'train_samples': split_info.get('train_samples'), 'val_samples': split_info.get('val_samples'), 'subset_stats': split_info.get('subset_stats')}
    run_inference(args, best_path)
    eval_summary = os.path.join('eval_results', proj, 'summary.csv')
    if not os.path.isfile(eval_summary):
        raise FileNotFoundError(f'seed={seed} 未找到测试汇总: {eval_summary}')
    seed_summary_copy = os.path.join(seed_dir, 'test_summary.csv')
    copy_summary(eval_summary, seed_summary_copy)
    test_scores = read_summary_csv(eval_summary)
    train_meta['test_summary_path'] = seed_summary_copy
    train_meta['test_scores'] = test_scores
    with open(os.path.join(seed_dir, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(train_meta, f, indent=2, ensure_ascii=False)
    return {'seed': seed, 'project': proj, 'test_scores': test_scores, 'train_meta': train_meta}
def write_per_seed_csv(path: str, all_runs: list[dict]) -> None:
    datasets = sorted(all_runs[0]['test_scores'].keys())
    fieldnames = ['seed', 'project', 'dataset', *METRICS]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in all_runs:
            for dataset in datasets:
                row = {'seed': run['seed'], 'project': run['project'], 'dataset': dataset}
                row.update({k: round_metric(v) for k, v in run['test_scores'][dataset].items()})
                writer.writerow(row)
def write_aggregate_csv(path: str, all_runs: list[dict]) -> None:
    datasets = sorted(all_runs[0]['test_scores'].keys())
    n = len(all_runs)
    fieldnames = ['dataset', 'metric', 'mean', 'std', 'n'] + [f"seed_{run['seed']}" for run in all_runs]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            for metric in METRICS:
                values = [run['test_scores'][dataset][metric] for run in all_runs]
                row = {'dataset': dataset, 'metric': metric, 'mean': round_metric(mean(values)), 'std': round_metric(stdev(values) if n > 1 else 0.0), 'n': n}
                for run, val in zip(all_runs, values):
                    row[f"seed_{run['seed']}"] = round_metric(val)
                writer.writerow(row)
def write_aggregate_report(path: str, all_runs: list[dict]) -> None:
    datasets = sorted(all_runs[0]['test_scores'].keys())
    n = len(all_runs)
    seeds = [run['seed'] for run in all_runs]
    lines = [f'Multi-seed experiment report', f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f'Seeds ({n}): {seeds}', '']
    val_sids = [run['train_meta'].get('best_val_sid') for run in all_runs if run['train_meta'].get('best_val_sid') is not None]
    if val_sids:
        m = mean(val_sids)
        s = stdev(val_sids) if len(val_sids) > 1 else 0.0
        detail = ', '.join((format_metric(v) for v in val_sids))
        lines.append('## Validation S_ID (model selection)')
        lines.append(f'  mean ± std: {format_metric(m)} ± {format_metric(s)}  ({detail})')
        lines.append('')
    for dataset in datasets:
        lines.append(f'## {dataset}')
        for metric in METRICS:
            values = [run['test_scores'][dataset][metric] for run in all_runs]
            m = mean(values)
            s = stdev(values) if n > 1 else 0.0
            detail = ', '.join((f"s{run['seed']}={format_metric(v)}" for run, v in zip(all_runs, values)))
            lines.append(f'  {metric:8s}: {format_metric(m)} ± {format_metric(s)}  ({detail})')
        lines.append('')
    lines.append('## Overall (mean over datasets, then over seeds)')
    for metric in METRICS:
        per_seed_avg = []
        for run in all_runs:
            per_seed_avg.append(mean((run['test_scores'][ds][metric] for ds in datasets)))
        m = mean(per_seed_avg)
        s = stdev(per_seed_avg) if n > 1 else 0.0
        detail = ', '.join((f"s{run['seed']}={format_metric(v)}" for run, v in zip(all_runs, per_seed_avg)))
        lines.append(f'  {metric:8s}: {format_metric(m)} ± {format_metric(s)}  ({detail})')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
def main():
    args = parse_args()
    base_project = args.project
    out_dir = os.path.join(args.output_root, base_project)
    os.makedirs(out_dir, exist_ok=True)
    datasets = discover_datasets(args.test_path, args.datasets)
    if not datasets:
        raise FileNotFoundError(f'在 {args.test_path} 下未找到测试数据集')
    config = {'base_project': base_project, 'seeds': args.seeds, 'test_datasets': datasets, 'device': args.device, 'epoch': args.epoch, 'val_ratio': args.val_ratio, 'skip_train': args.skip_train, 'started_at': datetime.now().isoformat(timespec='seconds')}
    with open(os.path.join(out_dir, 'experiment_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    all_runs = []
    for seed in args.seeds:
        all_runs.append(run_one_seed(args, base_project, seed, out_dir))
    per_seed_csv = os.path.join(out_dir, 'per_seed_all.csv')
    aggregate_csv = os.path.join(out_dir, 'aggregate_mean_std.csv')
    report_txt = os.path.join(out_dir, 'aggregate_report.txt')
    write_per_seed_csv(per_seed_csv, all_runs)
    write_aggregate_csv(aggregate_csv, all_runs)
    write_aggregate_report(report_txt, all_runs)
    config['finished_at'] = datetime.now().isoformat(timespec='seconds')
    config['output_files'] = {'per_seed_all': per_seed_csv, 'aggregate_mean_std': aggregate_csv, 'aggregate_report': report_txt}
    with open(os.path.join(out_dir, 'experiment_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print('\n' + '#' * 70)
    print('多种子实验完成')
    print(f'汇总目录: {out_dir}')
    print(f'  - {per_seed_csv}')
    print(f'  - {aggregate_csv}')
    print(f'  - {report_txt}')
    print('#' * 70)
    with open(report_txt, encoding='utf-8') as f:
        print(f.read())
if __name__ == '__main__':
    main()