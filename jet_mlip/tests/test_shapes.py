"""jet_mlip.shapes.emit_line round-trips every shape; configs.selected() family modes."""
import pytest

from jet_mlip import configs, tables
from jet_mlip.shapes import emit_line


def test_emit_line_round_trips_every_shape():
    for cfg in configs.selected('all'):
        ns = {'tables': tables, 'cfgs': []}
        exec(emit_line(cfg, 'test').strip(), ns)
        assert [c.key for c in ns['cfgs']] == [cfg.key]


def test_selected_families_plus_user():
    assert [c.key for c in configs.selected('mace,nequip')] == [
        c.key for c in configs.mace_configs() + configs.nequip_configs() + configs.user_configs()]


def test_unknown_family_exits():
    with pytest.raises(SystemExit):
        configs.selected('bogus')
