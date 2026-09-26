import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import minalphafold.trainer as trainer
from tests.test_data_pipeline import write_processed_cache
from tests.test_trainer import make_processed_cache_dirs


def configs(tmp_path):
    features, labels = make_processed_cache_dirs(tmp_path)
    data = trainer.DataConfig(processed_features_dir=features, processed_labels_dir=labels,
                              val_fraction=0.0, crop_size=4, msa_depth=2, extra_msa_depth=1, max_templates=1)
    training = trainer.TrainingConfig(device='cpu', seed=19, n_cycles=2, ema_decay=0.9)
    return trainer.load_model_config('tiny'), data, training


def test_epoch_resume_exactly_matches_uninterrupted_random_training(tmp_path):
    model, data, training = configs(tmp_path)
    uninterrupted, history = trainer.fit(model, data, replace(training, epochs=2))
    checkpoint = tmp_path / 'resume.pt'
    trainer.fit(model, data, replace(training, epochs=1, latest_checkpoint_path=checkpoint))
    resumed, resumed_history = trainer.fit(model, data, replace(training, epochs=2, resume_from_checkpoint=checkpoint))
    assert resumed_history == history
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in uninterrupted.state_dict().items())


def test_resume_rejects_changed_dataset_and_scientific_settings(tmp_path):
    model, data, training = configs(tmp_path)
    checkpoint = tmp_path / 'resume.pt'
    trainer.fit(model, data, replace(training, latest_checkpoint_path=checkpoint))
    with pytest.raises(ValueError, match='Resume checkpoint'):
        trainer.fit(model, data, replace(training, epochs=2, learning_rate=0.2, resume_from_checkpoint=checkpoint))
    write_processed_cache(Path(data.processed_features_dir), Path(data.processed_labels_dir), '1abc_A', 'GGAAA', include_templates=True)
    with pytest.raises(ValueError, match='Resume checkpoint'):
        trainer.fit(model, data, replace(training, epochs=2, resume_from_checkpoint=checkpoint))
    with pytest.raises(FileExistsError):
        trainer.fit(model, data, replace(training, latest_checkpoint_path=checkpoint))


def test_resume_rejects_changed_chemical_reference_resource(tmp_path, monkeypatch):
    model, data, training = configs(tmp_path)
    # Use an isolated source inventory so this test never edits shared chemistry.
    package = Path(trainer.__file__).parent
    shadow = tmp_path / 'source'
    shadow.mkdir()
    for name in ('trainer.py', 'stereo_chemical_props.txt'):
        (shadow / name).write_bytes((package / name).read_bytes())
    monkeypatch.setattr(trainer, '__file__', str(shadow / 'trainer.py'))
    checkpoint = tmp_path / 'chemistry.pt'
    trainer.fit(model, data, replace(training, latest_checkpoint_path=checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    assert 'stereo_chemical_props.txt' in payload['resume_contract']['implementation']
    resource = shadow / 'stereo_chemical_props.txt'
    resource.write_bytes(resource.read_bytes() + b'\nchanged scientific resource\n')
    with pytest.raises(ValueError, match='Resume checkpoint'):
        trainer.fit(model, data, replace(training, epochs=2, resume_from_checkpoint=checkpoint))


def test_partial_accumulation_clips_before_averaging_and_keeps_remainder(tmp_path, monkeypatch):
    model, data, training = configs(tmp_path)
    write_processed_cache(Path(data.processed_features_dir), Path(data.processed_labels_dir), '3abc_A', 'GAGAA', include_templates=False)
    class Scalar(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
        def forward(self, **kwargs):
            return {'value': self.weight.expand(kwargs['aatype'].shape[0])}
    class Loss(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, **kwargs):
            return kwargs['value'] * 2.0
    monkeypatch.setattr(trainer, 'AlphaFold2', Scalar)
    monkeypatch.setattr(trainer, 'AlphaFoldLoss', Loss)
    monkeypatch.setattr(trainer, 'loss_inputs_from_batch', lambda batch, output: output)
    monkeypatch.setattr(trainer, 'build_optimizer', lambda model, config: torch.optim.SGD(model.parameters(), lr=1.0))
    trained, history = trainer.fit(model, data, replace(training, learning_rate=1, grad_accum_steps=2, grad_clip_norm=0.1, ema_decay=None))
    assert history[-1]['global_samples'] == 3
    assert history[-1]['global_step'] == 2
    assert trained.weight.item() == pytest.approx(-0.2)


def test_exact_sample_cap_with_recycling_and_partial_batch(tmp_path):
    model, data, training = configs(tmp_path)
    _, history = trainer.fit(model, data, replace(training, epochs=3, batch_size=2, grad_accum_steps=3, max_samples=3))
    assert history[-1]['global_samples'] == 3
    assert history[-1]['global_step'] == 2


@pytest.mark.parametrize('change', [{'grad_accum_steps': 0}, {'learning_rate': float('nan')}, {'epochs': 0}, {'batch_size': 0}])
def test_invalid_training_settings_fail_before_outputs(tmp_path, change):
    model, data, training = configs(tmp_path)
    output = tmp_path / 'bad.pt'
    with pytest.raises(ValueError):
        trainer.fit(model, data, replace(training, latest_checkpoint_path=output, **change))
    assert not output.exists()


def test_nonfinite_loss_and_gradient_fail_closed():
    with pytest.raises(ValueError, match='Non-finite loss'):
        trainer.checked_loss(torch.tensor([float('nan')]))
    model = torch.nn.Linear(1, 1)
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    with pytest.raises(RuntimeError, match='non-finite'):
        trainer.clip_gradients(model, None)


@pytest.mark.parametrize("frozen", [False, True])
def test_processed_overfit_restricts_requested_chain_and_has_finite_cpu_output(tmp_path, frozen):
    from overfit_processed_chain import main
    features, labels = make_processed_cache_dirs(tmp_path)
    result = main(['--chain-id', '2xyz_A', '--processed-features-dir', str(features),
                   '--processed-labels-dir', str(labels), '--model-profile', 'tiny', '--device', 'cpu',
                   '--steps', '1', '--log-every', '1', '--crop-size', '5', '--msa-depth', '2',
                   '--extra-msa-depth', '1', '--max-templates', '1', '--out-dir', str(tmp_path / 'run')] + (['--freeze-crop-and-cluster'] if frozen else []))
    # The first sorted cached chain is different. This detects accidental use
    # of the unrestricted directory loader for the overfit target.
    truth = Path(result['ground_truth_pdb']).read_text()
    residue_names = [line[17:20] for line in truth.splitlines() if line.startswith('ATOM') and line[12:16].strip() == 'CA']
    assert residue_names == ['ALA', 'GLY', 'GLY', 'ALA', 'ALA']
    json.dumps(result, allow_nan=False, default=str)


@pytest.mark.parametrize('field,value', [('global_samples', None), ('global_step', -1), ('epoch', 1.5)])
def test_resume_rejects_corrupt_exposure_counters(tmp_path, field, value):
    model, data, training = configs(tmp_path)
    checkpoint = tmp_path / 'resume.pt'
    trainer.fit(model, data, replace(training, latest_checkpoint_path=checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='Checkpoint'):
        trainer.fit(model, data, replace(training, epochs=2, resume_from_checkpoint=checkpoint))


def test_resume_cannot_overwrite_another_run(tmp_path):
    model, data, training = configs(tmp_path)
    checkpoint_a, checkpoint_b = tmp_path / 'a.pt', tmp_path / 'b.pt'
    for path in (checkpoint_a, checkpoint_b):
        trainer.fit(model, data, replace(training, latest_checkpoint_path=path))
    original = checkpoint_b.read_bytes()
    with pytest.raises(FileExistsError, match='another run'):
        trainer.fit(model, data, replace(training, epochs=2, resume_from_checkpoint=checkpoint_a, latest_checkpoint_path=checkpoint_b))
    assert checkpoint_b.read_bytes() == original


def test_overflowing_optimizer_cannot_publish_checkpoint(tmp_path, monkeypatch):
    model, data, training = configs(tmp_path)
    original_step = torch.optim.Adam.step
    def overflowing_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        self.param_groups[0]['params'][0].data.fill_(float('inf'))
        return result
    monkeypatch.setattr(torch.optim.Adam, 'step', overflowing_step)
    checkpoint = tmp_path / 'bad.pt'
    with pytest.raises(ValueError, match='Non-finite model'):
        trainer.fit(model, data, replace(training, max_samples=1, latest_checkpoint_path=checkpoint))
    assert not checkpoint.exists()


def test_resume_restores_worker_and_validation_random_streams(tmp_path):
    model, data, training = configs(tmp_path)
    data = replace(data, val_fraction=0.5)
    training = replace(training, num_workers=1)
    uninterrupted, history = trainer.fit(model, data, replace(training, epochs=2))
    checkpoint = tmp_path / 'worker.pt'
    trainer.fit(model, data, replace(training, latest_checkpoint_path=checkpoint))
    resumed, resumed_history = trainer.fit(model, data, replace(training, epochs=2, resume_from_checkpoint=checkpoint))
    assert resumed_history == history
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in uninterrupted.state_dict().items())


def test_explicit_roles_drive_validation_and_bind_split_metadata(tmp_path):
    model, data, training = configs(tmp_path)
    train_path, val_path = tmp_path / 'train.json', tmp_path / 'val.json'
    train_path.write_text(json.dumps({'chains': [{'chain_id': '1abc_A', 'full_group_id': 'one'}]}))
    val_path.write_text(json.dumps({'chains': [{'chain_id': '2xyz_A', 'full_group_id': 'two'}]}))
    data = replace(data, train_chains_manifest=train_path, val_chains_manifest=val_path)
    checkpoint = tmp_path / 'roles.pt'
    _, history = trainer.fit(model, data, replace(training, latest_checkpoint_path=checkpoint))
    assert history[-1]['global_samples'] == 1
    assert 'val_loss' in history[-1]
    payload = torch.load(checkpoint, weights_only=False)
    assert payload['best_val_loss'] == history[-1]['val_loss']
    # Same IDs and tensors, changed group definition: resume must bind the
    # supplied ownership metadata, not only the selected chain set.
    val_path.write_text(json.dumps({'chains': [{'chain_id': '2xyz_A', 'full_group_id': 'three'}]}))
    with pytest.raises(ValueError, match='Resume checkpoint'):
        trainer.fit(model, data, replace(training, epochs=2, resume_from_checkpoint=checkpoint))
