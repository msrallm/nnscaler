#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.flags import CompileFlag
from nnscaler.parallel import ComputeConfig, parallelize

from tests.utils import replace_all_device_with
from tests.graph.schedule.test_interleaved_1f1b import Model, policy_1f1b, policy_1f1b_interleaved


def _gen(tempdir, policy, ngpus, skip):
    old = CompileFlag.skip_dummy_micros
    CompileFlag.skip_dummy_micros = skip
    try:
        m = Model()
        m.train()
        parallelize(
            m, {'x': torch.randn(2, 32)}, policy,
            ComputeConfig(ngpus, ngpus, use_end2end=True, pas_config=dict(n_micro_batches=4)),
            gen_savedir=tempdir, load_module=False, reuse='override',
        )
    finally:
        CompileFlag.skip_dummy_micros = old
    files = sorted((Path(tempdir) / '_parallel_modules').rglob('gencode*.py'))
    assert len(files) == ngpus
    return {f.name: f.read_text() for f in files}


def _function(src, name):
    start = src.index(f'def {name}(')
    nxt = src.find('\ndef ', start + 1)
    return src[start:] if nxt < 0 else src[start:nxt]


@replace_all_device_with('cpu')
@pytest.mark.parametrize('policy', [policy_1f1b, policy_1f1b_interleaved])
def test_skip_dummy_codegen(policy):
    with tempfile.TemporaryDirectory() as off_dir, tempfile.TemporaryDirectory() as on_dir:
        off = _gen(off_dir, policy, 2, False)
        on = _gen(on_dir, policy, 2, True)
        for fname, src in on.items():
            assert '_n_active' not in off[fname]
            compile(src, fname, 'exec')  # every rank's code stays valid python
            for fn in ('_train_step', '_infer_step'):
                if f'def {fn}(' not in src:
                    continue
                body = _function(src, fn)
                assert '_n_active = nnscaler.flags.RuntimeFlag.active_micros' in body
                # every p2p adapter call sits inside a micro-batch guard (8 spaces:
                # function body + guard block); reducers stay at function level
                for line in body.split('\n'):
                    if 'aexecute(model.adapter' in line:
                        assert line.startswith(' ' * 8), f'{fname}:{fn} unguarded p2p: {line.strip()}'
                    if 'aexecute(model.reducer' in line:
                        assert not line.startswith(' ' * 8), f'{fname}:{fn} guarded reducer: {line.strip()}'
                guards = set(re.findall(r'if (\d+) < _n_active:', body))
                assert guards == {'0', '1', '2', '3'}, f'{fname}:{fn} guards {sorted(guards)}'
