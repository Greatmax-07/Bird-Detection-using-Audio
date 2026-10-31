import numpy as np

from birdear.core import (POOLED_DIM, class_names_from_label_map, load_species, name_key, pool_segments,
                          recording_split, slugify)
from birdear.detect import DetectParams, detect_events, fuse


def test_pool_shape_and_norm():
    v = pool_segments(np.random.default_rng(0).normal(size=(7, 1024)))
    assert v.shape == (POOLED_DIM,)
    assert abs(np.linalg.norm(v) - 1) < 1e-5


def test_split_is_deterministic_and_disjoint():
    paths = [f"c{c}/{i}.wav" for c in range(4) for i in range(10)]
    labels = [c for c in range(4) for _ in range(10)]
    t1, v1 = recording_split(paths, labels)
    t2, v2 = recording_split(paths[::-1], labels[::-1])
    assert t1 == t2 and v1 == v2 and not (t1 & v1)


def test_species_file_aliases_and_slug():
    from birdear.core import load_species_entries
    names = load_species()
    assert len(names) == len(set(names)) > 50
    assert slugify("Tickell's Blue Flycatcher") == "Tickells_Blue_Flycatcher"
    assert next(e for e in load_species_entries() if e[0] == "Rock Dove") == ["Rock Dove", "Rock Pigeon"]
    assert name_key("Paradise-flycatcher") == name_key("paradise Flycatcher")
    assert name_key("Indian Grey Hornbill") == name_key("Indian Gray Hornbill")


def test_label_map_contiguous():
    assert class_names_from_label_map({"b": 1, "a": 0}) == ["a", "b"]


def _times(n):
    s = np.arange(n) * 3.0
    return np.stack([s, s + 3], axis=1)


def test_detect_two_species_hysteresis_and_gating():
    f = np.zeros((10, 3))
    f[2:7, 0] = [0.25, 0.3, 0.8, 0.3, 0.25]      # species 0: strong core with weak shoulders
    f[8, 1] = 0.3                                # species 1: never reaches t_high -> ignored
    f[0:2, 2] = 0.9                              # species 2: separate strong event
    f[8:10, 2] = 0.9                             # species 2: a second event (gap of 6 s)
    ev = detect_events(_times(10), f, DetectParams(t_high=0.4, t_low=0.2, merge_gap=3.0))
    assert set(ev) == {0, 2}
    assert len(ev[0]) == 1 and ev[0][0]["start_time"] == 6.0 and ev[0][0]["end_time"] == 21.0
    assert len(ev[2]) == 2


def test_isolated_weak_segment_is_dropped():
    f = np.zeros((12, 1))
    f[1] = 0.9
    f[10] = 0.25                                 # far from the strong part, below t_high
    ev = detect_events(_times(12), f, DetectParams())
    assert len(ev[0]) == 1


def test_fuse_uses_head_only_without_birdnet():
    ph, bn = np.full((2, 2), 0.2), np.full((2, 2), 0.8)
    out = fuse(ph, bn, np.array([True, False]), 0.5)
    assert np.allclose(out[:, 0], 0.5) and np.allclose(out[:, 1], 0.2)
