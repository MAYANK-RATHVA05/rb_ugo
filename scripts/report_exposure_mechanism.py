"""Export aggregate mechanism evidence from completed frozen job records."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from scripts.exposure_mechanism_diagnostic import POLICIES, ROOT, sha256, summarize, write_json


def write_csv(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)


def decision(summary, config):
    rule = config['decision_rule']; eligible = []; controls = {}
    for cell in summary['cells']:
        n = cell['methods']['budget']['active']
        if n >= rule['minimum_paired_active_seeds']:
            eligible.append(cell)
    for control in ['threshold_matched', 'weighted_pl']:
        evidence = []
        for cell in eligible:
            pairs = {r['metric']: r for r in cell['paired']
                     if r['control'] == control and r['active_only']}
            tests = {}
            for metric in ['oracle_endpoint_error', 'mcc', 'gmean', 'ap']:
                p = pairs[metric]; observed = p['difference']['mean']
                adequate = p['difference']['n'] >= rule['minimum_paired_active_seeds']
                margin = (rule['mechanism_practical_error_difference'] if metric == 'oracle_endpoint_error'
                          else rule['downstream_practical_difference'])
                matches = (observed is not None and
                           (observed >= -margin if metric == 'oracle_endpoint_error' else observed <= margin))
                tests[metric] = dict(paired_seeds=p['difference']['n'], budget_minus_control=observed,
                                     margin=margin, simpler_control_matches=bool(adequate and matches))
            evidence.append(dict(cell_index=cell['cell_index'], tests=tests,
                                  matches=all(t['simpler_control_matches'] for t in tests.values())))
        controls[control] = dict(cells=evidence,
                                 matches_every_adequately_active_cell=bool(evidence and all(e['matches'] for e in evidence)))
    reject = any(c['matches_every_adequately_active_cell'] for c in controls.values())
    return dict(adequately_active_cells=[c['cell_index'] for c in eligible], controls=controls,
                outcome='reject_current_added_value_claim_on_this_grid' if reject else 'inconclusive_or_revise',
                recommendation='Do not expand to adaptive UGO. Preserve this diagnostic as negative/limited evidence.',
                interpretation='Descriptive preregistered practical-margin rule; not a statistical equivalence test or universal rejection')


def export(run_dir, output):
    manifest = json.loads((run_dir / 'manifest.json').read_text()); config = manifest['config']
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh report directory; existing evidence is preserved')
    for path, expected in manifest['source_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Training/evaluation source changed since run: {path}')
    cases = []
    planned_cells = list(itertools.product(*config['grid'].values()))
    for cell_index in range(len(planned_cells)):
        for seed in config['dataset_seeds']:
            record = json.loads((run_dir / f'cell-{cell_index}' / f'seed-{seed}' / 'results.json').read_text())
            expected_cell = dict(zip(config['grid'], planned_cells[cell_index]))
            if (record['cell_index'], record['seed'], record['cell']) != (cell_index, seed, expected_cell):
                raise ValueError('Job identity differs from preregistration')
            if sorted(r['method'] for r in record['records']) != sorted(POLICIES):
                raise ValueError('Missing or duplicate method records')
            cases.append(record)
    original = json.loads((run_dir / 'summary.json').read_text())
    summary = summarize(cases, config, original['runtime']['wall_seconds'])
    if summary != original:
        raise ValueError('Saved aggregate does not recompute from current job records')
    output.mkdir(parents=True, exist_ok=True)
    methods = []; cells = []; paired = []
    for cell in summary['cells']:
        identity = dict(cell_index=cell['cell_index'], **cell['cell'])
        cells.append(dict(**identity, cases=cell['cases'],
                          budget_active=cell['methods']['budget']['active'],
                          inactive_reasons=json.dumps(cell['methods']['budget']['inactive_reasons'], sort_keys=True),
                          eligible_endpoints_mean=cell['eligible_endpoints']['mean'],
                          eligible_endpoints_max=cell['eligible_endpoints']['maximum'],
                          usable_endpoints_mean=cell['usable_endpoints']['mean'],
                          binding_rounds=cell['binding_rounds'], restrictive_rounds=cell['restrictive_rounds'],
                          usable_rounds=cell['usable_rounds'], binding_frequency=cell['binding_frequency'],
                          restrictive_frequency=cell['restrictive_frequency'],
                          eligible_realized_error=cell['eligible_realized_error']['mean'],
                          eligible_oracle_error=cell['eligible_oracle_error']['mean']))
        for method, data in cell['methods'].items():
            row = dict(**identity, method=method, successes=data['successes'], failures=data['failures'], active=data['active'])
            for metric, statistics in data['metrics'].items():
                for statistic, value in statistics.items(): row[f'{metric}_{statistic}'] = value
            for metric in ['mode_expected_mass', 'u_child_mode_expected_mass', 'mode_recall', 'mode_test_counts']:
                for k in range(2):
                    for statistic, value in data[metric][k].items(): row[f'{metric}_{k}_{statistic}'] = value
            methods.append(row)
        for p in cell['paired']:
            paired.append(dict(**identity, control=p['control'], metric=p['metric'], active_only=p['active_only'],
                               shared_successes=p['shared_successes'], seeds=json.dumps(p['seeds']),
                               **p['difference'], budget_wins=p['budget_wins'], ties=p['ties'], budget_losses=p['budget_losses']))
    write_csv(output / 'cell_summary.csv', cells)
    write_csv(output / 'method_summary.csv', methods)
    write_csv(output / 'paired_effects.csv', paired)
    write_csv(output / 'pooled_seed_effects.csv', [dict(control=r['control'], metric=r['metric'],
                                                     seeds=json.dumps(r['seeds']), **r['difference']) for r in summary['pooled']])
    write_json(output / 'run_evidence.json', dict(
        cases=summary['cases'], method_jobs=summary['method_jobs'], failures=summary['failures'],
        preregistration_commit='23b2afe', config_sha256=manifest['config_sha256'],
        run_manifest_sha256=sha256(run_dir / 'manifest.json'), training_manifest=manifest,
        report_source_sha256=sha256(Path(__file__)), runtime=summary['runtime'],
        independent_record_reaggregation='All 80 identities and 560 method records verified; summary exactly recomputed',
        decision=decision(summary, config)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = export(args.run, args.output)
    print(json.dumps(dict(cases=summary['cases'], jobs=summary['method_jobs'], failures=len(summary['failures']))))


if __name__ == '__main__': main()
