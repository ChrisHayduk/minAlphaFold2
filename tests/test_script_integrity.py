import json
from pathlib import Path

import numpy as np
import pytest
import torch
from download_openproteinset import download_subset, expand_duplicate_alignments
from filter_openproteinset import build_manifest, load_cluster_tsv
from overfit_single_pdb import main as overfit_pdb
from overfit_single_pdb import parse_pdb
from preprocess_openproteinset import project_to_query, write_cache_pair

from minalphafold.data import ProcessedOpenProteinSetDataset
from minalphafold.pdbio import write_atom14_pdb
from tests.test_trainer import make_processed_cache_dirs


def test_cache_pair_generation_detects_interrupted_replacement(tmp_path):
    features, labels = make_processed_cache_dirs(tmp_path)
    feature_path, label_path = features / '1abc_A.npz', labels / '1abc_A.npz'
    with np.load(feature_path) as source:
        feature_arrays = dict(source)
    with np.load(label_path) as source:
        label_arrays = dict(source)
    old_label = label_path.read_bytes()
    write_cache_pair(feature_path, label_path, feature_arrays, label_arrays)
    dataset = ProcessedOpenProteinSetDataset(features, labels, split='all')
    assert dataset[0]['chain_id'] == '1abc_A'
    label_path.write_bytes(old_label)
    with pytest.raises(ValueError, match='generations'):
        dataset[0]


def test_projection_never_assigns_wrong_residue_atom_slots():
    positions = np.ones((2, 14, 3), dtype=np.float32)
    mask = np.ones((2, 14), dtype=np.float32)
    projected, projected_mask, _ = project_to_query('AG', 'AA', positions, mask)
    assert projected_mask[0].sum() == 14
    assert projected_mask[1].sum() == 0
    assert projected[1].sum() == 0


def test_relative_duplicate_symlink_resolves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path('data/roda_pdb')
    (root / '1abc_A').mkdir(parents=True)
    (root / '1abc_A/query.a3m').write_text('>q\nAG\n')
    groups = Path('duplicates.txt')
    groups.write_text('1abc_A 2xyz_A\n')
    expand_duplicate_alignments(root, groups)
    assert (root / '2xyz_A/query.a3m').read_text() == '>q\nAG\n'


def test_missing_subset_asset_cannot_report_success(tmp_path, monkeypatch):
    import download_openproteinset
    monkeypatch.setattr(download_openproteinset, 'download_url', lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match='incomplete'):
        download_subset(tmp_path, ['1abc_A'], 'query.a3m', 'hits.hhr', skip_templates=True, dry_run=False)


def test_cluster_duplicate_and_malformed_members_fail(tmp_path):
    path = tmp_path / 'clusters.tsv'
    for contents in ('a\tb\na\tb\n', 'a\tb\nc\tb\n', 'badrow\n'):
        path.write_text(contents)
        with pytest.raises(ValueError):
            load_cluster_tsv(path)


def test_missing_resolution_is_explicit_json_null(tmp_path):
    features, labels = make_processed_cache_dirs(tmp_path)
    label_path = labels / '1abc_A.npz'
    with np.load(label_path) as source:
        arrays = dict(source)
    arrays['resolution'] = np.asarray(float('nan'))
    np.savez_compressed(label_path, **arrays)
    manifest = build_manifest(processed_features_dir=features, processed_labels_dir=labels,
        max_resolution=9, max_single_aa_fraction_threshold=0.8, min_length=1, mmseqs_cluster_tsv=None)
    assert manifest['chains'][0]['resolution'] is None
    json.dumps(manifest, allow_nan=False)


def pdb_fixture(tmp_path):
    path = tmp_path / 'input.pdb'
    aatype = torch.tensor([0, 7, 0])
    coords = torch.zeros(3, 14, 3)
    for i in range(3):
        coords[i, :4] = torch.tensor([[i * 3.8, 0., 0.], [i * 3.8 + 1., 1., 0.],
                                     [i * 3.8 + 2.3, 0., 0.], [i * 3.8 + 2.8, -1., 0.]])
    mask = torch.zeros(3, 14)
    mask[:, :4] = 1
    write_atom14_pdb(path, aatype, coords, mask)
    return path


def test_single_pdb_tiny_cpu_and_stale_output_rejection(tmp_path):
    path = pdb_fixture(tmp_path)
    argv = ['--pdb', str(path), '--steps', '1', '--log-every', '1', '--model-profile', 'tiny',
            '--device', 'cpu', '--n-cycles', '2', '--out-dir', str(tmp_path / 'run')]
    result = overfit_pdb(argv)
    assert result['best']['step'] == 1
    json.dumps(result, allow_nan=False, default=str)
    with pytest.raises(FileExistsError):
        overfit_pdb(argv)


def test_pdb_ambiguity_cannot_silently_merge_models_or_chains(tmp_path):
    path = pdb_fixture(tmp_path)
    original = path.read_text()
    path.write_text('MODEL        1\n' + original + 'ENDMDL\nMODEL        2\n' + original)
    with pytest.raises(ValueError, match='one model'):
        parse_pdb(path)
    lines = original.splitlines()
    line = next(line for line in lines if line.startswith('ATOM'))
    lines.append(line[:21] + 'B' + line[22:])
    path.write_text('\n'.join(lines))
    with pytest.raises(ValueError, match='one chain'):
        parse_pdb(path)


def test_relaxation_invalid_limits_rejected_without_optional_runtime(tmp_path):
    from relax_pdb import relax_pdb
    with pytest.raises(ValueError, match='iteration'):
        relax_pdb(tmp_path / 'in.pdb', tmp_path / 'out.pdb', max_rounds=0)


def test_modal_resume_honors_checkpoint_directory_and_flushes_on_failure(tmp_path, monkeypatch):
    # Compile the actual wrapper function without importing/initializing the
    # optional Modal SDK or requesting any remote resources.
    import ast
    import sys
    from types import SimpleNamespace

    import train_af2
    script = Path(__file__).resolve().parents[1] / 'scripts/modal_train_af2.py'
    node = next(node for node in ast.parse(script.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'run_train')
    node.decorator_list = []
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    calls, commits = [], []
    namespace = {'Path': Path, 'os': SimpleNamespace(chdir=lambda path: None),
                 'checkpoints_volume': SimpleNamespace(commit=lambda: commits.append(True))}
    exec(compile(tree, str(script), 'exec'), namespace)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(train_af2, 'main', lambda argv: calls.append(argv))
    checkpoint = tmp_path / 'initial_latest.pt'
    checkpoint.touch()
    argv = ['--checkpoint-dir', str(tmp_path)]
    namespace['run_train'](argv, auto_resume_stage='initial')
    assert calls == [[*argv, '--resume', str(checkpoint)]]
    assert len(commits) == 1
    def failed(argv):
        raise RuntimeError('bounded fixture')
    monkeypatch.setattr(train_af2, 'main', failed)
    with pytest.raises(RuntimeError, match='bounded fixture'):
        namespace['run_train'](argv, auto_resume_stage='initial')
    assert len(commits) == 2


def test_stage_budget_counts_only_filtered_training_population(tmp_path, monkeypatch):
    from dataclasses import replace

    import train_af2

    from minalphafold.trainer import load_training_protocol
    features, labels = make_processed_cache_dirs(tmp_path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'chains': [{'chain_id': '2xyz_A', 'accepted': True}]}))
    protocol = load_training_protocol('alphafold2')
    protocol.initial = replace(protocol.initial, total_samples=3, crop_size=5)
    monkeypatch.setattr(train_af2, 'load_training_protocol', lambda _: protocol)
    captured = []
    monkeypatch.setattr(train_af2, 'fit', lambda **kwargs: captured.append(kwargs))
    train_af2.main(['--stage', 'initial', '--model-config', 'tiny', '--checkpoint-dir', str(tmp_path / 'run'),
                   '--processed-features-dir', str(features), '--processed-labels-dir', str(labels),
                   '--chains-manifest', str(manifest)])
    assert captured[0]['training_config'].epochs == 3
    assert captured[0]['training_config'].max_samples == 3
    with pytest.raises(ValueError, match='fine-tune stage'):
        train_af2.main(['--stage', 'initial', '--checkpoint-dir', str(tmp_path / 'other'), '--init-from', 'wrong.pt'])


def test_templates_exclude_other_chains_of_target_and_negative_offsets(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import preprocess_openproteinset as preprocess
    chain_dir = tmp_path / 'target'
    (chain_dir / 'hhr').mkdir(parents=True)
    (chain_dir / 'hhr/hits.hhr').touch()
    (tmp_path / '1abc.cif').touch()
    (tmp_path / '2abc.cif').touch()
    monkeypatch.setattr(preprocess, 'parse_hhr_hits', lambda path: [
        preprocess.HHRHit('1abc', 'B', ((0, 0),)),
        preprocess.HHRHit('2abc', 'A', ((-1, 1), (1, -1), (0, 0))),
    ])
    calls = []
    def extract(path, pdb_id, chain_id):
        calls.append(pdb_id)
        return SimpleNamespace(sequence='AG', aatype=np.array([0, 7]), release_date='2000-01-01', polymer_sequence_complete=True,
                               atom14_positions=np.ones((2, 14, 3)), atom14_mask=np.ones((2, 14)))
    monkeypatch.setattr(preprocess, 'extract_chain_atoms', extract)
    result = preprocess.template_features(chain_dir, tmp_path, target_pdb_id='1abc', target_chain_id='A',
        query_sequence='AG', query_length=2, template_hhr_name='hits.hhr', max_templates=2, max_template_date='2020-01-01')
    assert calls == ['2abc']
    assert result['template_atom14_mask'][0, 0].sum() == 14
    assert result['template_atom14_mask'][0, 1].sum() == 0
