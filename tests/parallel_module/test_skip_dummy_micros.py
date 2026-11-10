#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import nnscaler
from nnscaler.flags import CompileFlag, RuntimeFlag
from nnscaler.parallel import ComputeConfig, parallelize

from tests.launch_torchrun import torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0
from tests.graph.schedule.test_interleaved_1f1b import Model, policy_1f1b, policy_1f1b_interleaved


def _build(model, data, policy, skip, tag):
    old = CompileFlag.skip_dummy_micros
    CompileFlag.skip_dummy_micros = skip
    try:
        with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'skip_dummy_{tag}_{PYTEST_RUN_ID}') as tempdir:
            pm = parallelize(
                model, {'x': data}, policy,
                ComputeConfig(2, 2, use_end2end=True, pas_config=dict(n_micro_batches=4)),
                gen_savedir=tempdir, reuse='override', instance_name=tag,
            )
            gencode = (Path(tempdir) / '_parallel_modules').rglob(f'gencode{torch.distributed.get_rank()}.py')
            src = next(gencode).read_text()
    finally:
        CompileFlag.skip_dummy_micros = old
    pm.to(torch.cuda.current_device())
    return pm, src


def _step(pm, samples, is_dummy, active):
    for p in pm.parameters():
        p.grad = None
    RuntimeFlag.active_micros = active
    try:
        outputs = pm.train_step(samples, is_dummy_batch=is_dummy)
    finally:
        RuntimeFlag.active_micros = None
    grads = {n: p.grad.detach().clone() for n, p in pm.named_parameters() if p.grad is not None}
    return outputs, grads


def _assert_same_grads(a, b):
    assert a.keys() == b.keys() and len(a) > 0
    for k in a:
        assert torch.equal(a[k], b[k]), f'{k}: max|diff| {(a[k] - b[k]).abs().max().item()}'


def worker_skip_dummy(policy_name):
    import faulthandler, sys
    faulthandler.dump_traceback_later(240, exit=True, file=sys.stderr)
    nnscaler.init()
    torch.manual_seed(0)
    policy = {'1f1b': policy_1f1b, '1f1b_interleaved': policy_1f1b_interleaved}[policy_name]
    device = torch.cuda.current_device()
    data = torch.randn(2, 32, device=device)
    base = Model()
    base.train()

    torch.manual_seed(1)
    real = [torch.randn(2, 32, device=device) for _ in range(2)]
    # trailing dummies clone a real sample (verl's make_dummy convention); loss x0
    samples = real + [real[0].clone(), real[0].clone()]
    is_dummy = [False, False, True, True]

    pm_off, src_off = _build(base, data, policy, False, f'{policy_name}_off')
    pm_on, src_on = _build(base, data, policy, True, f'{policy_name}_on')
    assert '_n_active' not in src_off
    assert '_n_active' in src_on and 'if 3 < _n_active:' in src_on

    out_ref, g_ref = _step(pm_off, samples, is_dummy, None)
    _, g_ref2 = _step(pm_off, samples, is_dummy, None)
    _assert_same_grads(g_ref, g_ref2)  # reference is deterministic step to step
    # alternate skipped / full steps: no state may leak from a skipped step
    for active in (2, None, 2, None):
        out, g = _step(pm_on, samples, is_dummy, active)
        _assert_same_grads(g_ref, g)
        if active == 2:
            out_skip = out

    rank = torch.distributed.get_rank()
    print(f'[rank{rank}] {policy_name}: skip outputs {[None if o is None else "t" for o in out_skip]}')
    assert out_skip[2] is None and out_skip[3] is None
    for i in range(2):
        assert torch.equal(out_skip[i], out_ref[i]), i


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('policy_name', ['1f1b', '1f1b_interleaved'])
def test_skip_dummy_micros(policy_name):
    torchrun(2, worker_skip_dummy, policy_name)
