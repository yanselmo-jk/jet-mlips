"""patch_tool: pick the patch file that dry-runs cleanly, apply, refuse twice, reverse."""
import pathlib
import subprocess

import pytest

from common import patch_tool


@pytest.fixture
def fake_pkg(tmp_path, monkeypatch):
    root = tmp_path / 'site-packages'
    (root / 'pkgx').mkdir(parents=True)
    (root / 'pkgx' / 'a.py').write_text('x = 1\n')
    new = tmp_path / 'new'
    (new / 'pkgx').mkdir(parents=True)
    (new / 'pkgx' / 'a.py').write_text('x = 2\n')
    (new / 'pkgx' / 'marker.py').write_text('patched = True\n')
    patches = tmp_path / 'patches'
    patches.mkdir()
    diff = subprocess.run(['diff', '-ruN', 'site-packages', 'new'], cwd=tmp_path, capture_output=True, text=True).stdout
    (patches / 'pkgx-9.9.patch').write_text(diff.replace('site-packages/', 'a/').replace('new/', 'b/'))
    (patches / 'pkgx-1.0.patch').write_text(diff.replace('x = 1', 'x = 0').replace('site-packages/', 'a/').replace('new/', 'b/'))
    monkeypatch.setitem(patch_tool.PKGS, 'pkgx', patch_tool.Pkg('pkgx', 'pkgx/marker.py', 'pkgx'))
    monkeypatch.setattr(patch_tool, 'PATCHES', str(patches))
    return root


def test_check_apply_refuse_reverse(fake_pkg, capsys):
    root = str(fake_pkg)
    assert patch_tool.main(['pkgx', '--root', root, '--check']) == 0
    assert 'pkgx-9.9.patch' in capsys.readouterr().out      # the 1.0 variant does not apply
    assert (fake_pkg / 'pkgx' / 'a.py').read_text() == 'x = 1\n'
    assert patch_tool.main(['pkgx', '--root', root]) == 0
    assert (fake_pkg / 'pkgx' / 'a.py').read_text() == 'x = 2\n'
    assert (fake_pkg / 'pkgx' / 'marker.py').exists()
    assert patch_tool.main(['pkgx', '--root', root]) == 0        # refuses, does not double-apply
    assert 'already patched' in capsys.readouterr().out
    assert patch_tool.main(['pkgx', '--root', root, '--reverse']) == 0
    assert (fake_pkg / 'pkgx' / 'a.py').read_text() == 'x = 1\n'
    assert not (fake_pkg / 'pkgx' / 'marker.py').exists()


def test_no_patch_applies(fake_pkg, capsys):
    (fake_pkg / 'pkgx' / 'a.py').write_text('x = 7\n')
    assert patch_tool.main(['pkgx', '--root', str(fake_pkg), '--check']) == 1
    assert 'no shipped patch applies' in capsys.readouterr().out


def test_reverse_uses_the_applied_copy(fake_pkg, tmp_path):
    """Regenerating the shipped patch after applying it must not break --reverse."""
    root = str(fake_pkg)
    assert patch_tool.main(['pkgx', '--root', root]) == 0
    p = pathlib.Path(patch_tool.PATCHES) / 'pkgx-9.9.patch'
    p.write_text(p.read_text().replace('x = 2', 'x = 3'))   # shipped file moved on
    assert patch_tool.main(['pkgx', '--root', root, '--reverse']) == 0
    assert (fake_pkg / 'pkgx' / 'a.py').read_text() == 'x = 1\n'


def test_reverse_without_record(fake_pkg):
    """Trees patched before the record existed (old jet_mlip.patch_sevenn): --reverse finds the variant by dry run."""
    root = str(fake_pkg)
    assert patch_tool.main(['pkgx', '--root', root]) == 0
    for f in (fake_pkg / 'pkgx').glob('.mlip_accel_patch*'):
        f.unlink()
    assert patch_tool.main(['pkgx', '--root', root, '--reverse']) == 0
    assert (fake_pkg / 'pkgx' / 'a.py').read_text() == 'x = 1\n'
