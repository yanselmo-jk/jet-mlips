"""jet_mlip.deploy forwards to each family's CLI without importing the family packages."""
import types

import pytest

from jet_mlip import deploy


@pytest.fixture
def captured(monkeypatch):
    """Fake module: records the imported path and the argv the family main got."""
    seen = {}

    def fake_main(argv):
        seen['argv'] = argv

    def fake_import(name):
        seen['module'] = name
        return types.SimpleNamespace(main=fake_main)

    monkeypatch.setattr(deploy.importlib, 'import_module', fake_import)
    return seen


@pytest.mark.parametrize('family,path,argv', [
    ('sevenn', 'jet_mlip.sevenn_mliap', ['7net-0', 'out.pt', '--no-jet']),
    ('mace', 'jet_mace.deploy', ['medium', 'out.pt']),
    ('nequip', 'jet_nequip.deploy', ['nequip.net:mir-group/NequIP-OAM-L:0.1', 'out.nequip.lmp.pt']),
    ('nequix', 'jet_nequix.deploy', ['nequix-mp-1.pt', 'out.pt']),
    ('equflash', 'jet_equflash.deploy', ['equflash_oam_30M.pt', 'out.pt']),
])
def test_family_dispatches(captured, family, path, argv):
    assert deploy.main([family, *argv]) == 0
    assert captured == {'module': path, 'argv': argv}


@pytest.mark.parametrize('family,model,output', [
    ('mace', 'medium', 'jet_deployed_medium.pt'),
    ('nequip', 'nequip.net:mir-group/NequIP-OAM-L:0.1', 'jet_deployed_NequIP-OAM-L-0.1.nequip.lmp.pt'),
    ('nequix', '/x/nequix-mp-1.pt', 'jet_deployed_nequix-mp-1.pt'),
    ('sevenn', '7net-0', 'jet_deployed_7net-0.pt'),
])
def test_default_output(captured, family, model, output):
    assert deploy.main([family, model]) == 0
    assert captured['argv'] == [model, output]


def test_default_output_before_options(captured):
    assert deploy.main(['mace', 'medium', '--no-cueq']) == 0
    assert captured['argv'] == ['medium', 'jet_deployed_medium.pt', '--no-cueq']


@pytest.mark.parametrize('argv', [[], ['bogus']])
def test_bad_usage_returns_2_without_importing(captured, argv):
    assert deploy.main(argv) == 2
    assert captured == {}


@pytest.mark.parametrize('flag', ['-h', '--help'])
def test_help_returns_0(captured, flag):
    assert deploy.main([flag]) == 0
    assert captured == {}
