from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.state_compilation.retrieve_state import generate_uniform_state


class TestRetStates(TestCase):
    def test_generate_uniform_state(self):
        try:
            generate_uniform_state([2, 3], "wrong")
        except Exception:
            assert True
        ghz = generate_uniform_state([2, 3], "ghz")
        w = generate_uniform_state([2, 3], "qudit-w-state")
        wemb = generate_uniform_state([2, 3], "embedded-w-state")
        assert np.allclose(ghz, np.array([np.sqrt(2) / 2, 0, 0, 0, np.sqrt(2) / 2, 0]))
        assert np.allclose(wemb, np.array([0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0, 0]))
        assert np.allclose(w, np.array([0, 0.57735, 0.57735, 0.57735, 0, 0]))

    def test_verify_normalized_state(self):
        dimension = 10
        non_normalized_vector = np.random.rand(dimension)
        non_normalized_vector = np.abs(non_normalized_vector) ** 2
        normalized_vector = non_normalized_vector / np.linalg.norm(non_normalized_vector)
        normalized_vector = np.abs(normalized_vector) ** 2
        assert not np.isclose(np.sum(non_normalized_vector), 1.0, atol=1e-08)
        assert np.isclose(np.sum(normalized_vector), 1.0, atol=1e-08)
