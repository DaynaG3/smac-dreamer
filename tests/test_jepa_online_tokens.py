import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smacdreamer.jepa.online_tokens import JEPATokenSpec, encode_state_vector


def test_encode_state_vector_matches_dataset_layout():
    spec = JEPATokenSpec(
        n_agents=2,
        n_enemies=1,
        max_agents=3,
        max_enemies=2,
        max_actions=4,
        ally_state_feat_size=3,
        enemy_state_feat_size=2,
        dynamic_token_dim=3,
        entity_static_feat_size=2,
        static_dim=5,
        token_dim=5,
        ally_has_shields=False,
        enemy_has_shields=False,
        num_unit_types=0,
    )
    state = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    entity_static = np.arange(spec.entities * 2, dtype=np.float32).reshape(spec.entities, 2)
    tokens, mask, slot = encode_state_vector(state, spec, entity_static)
    assert tokens.shape == (5, 5)
    np.testing.assert_allclose(tokens[0, :3], [1, 2, 3])
    np.testing.assert_allclose(tokens[1, :3], [4, 5, 6])
    np.testing.assert_allclose(tokens[3, :2], [7, 8])
    np.testing.assert_allclose(tokens[:, 3:5], entity_static)
    assert mask.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert slot.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
