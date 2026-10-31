import numpy as np

from birdear.core import (POOLED_DIM, class_names_from_label_map, load_species, pool_segments,
                          recording_split, slugify, window_ranges)


def test_pool_shape_and_norm():
    v = pool_segments(np.random.default_rng(0).normal(size=(7, 1024)))
    assert v.shape == (POOLED_DIM,)
    assert abs(np.linalg.norm(v) - 1) < 1e-5


def test_pool_single_segment_has_zero_std():
    v = pool_segments(np.ones((1, 1024)))
    assert np.allclose(v[2048:], 0)


def test_window_ranges_cover_everything():
    for n in range(1, 40):
        ranges = window_ranges(n, 5, 2)
        covered = set()
        for a, b in ranges:
            assert 0 <= a < b <= n
            covered.update(range(a, b))
        assert covered == set(range(n)), n


def test_split_is_deterministic_and_disjoint():
    paths = [f"c{c}/{i}.wav" for c in range(4) for i in range(10)]
    labels = [c for c in range(4) for _ in range(10)]
    t1, v1 = recording_split(paths, labels)
    t2, v2 = recording_split(paths[::-1], labels[::-1])
    assert t1 == t2 and v1 == v2 and not (t1 & v1)


def test_species_file_and_slug():
    names = load_species()
    assert len(names) == len(set(names)) > 50
    assert slugify("Tickell's Blue Flycatcher") == "Tickells_Blue_Flycatcher"


def test_label_map_contiguous():
    assert class_names_from_label_map({"b": 1, "a": 0}) == ["a", "b"]


def test_species_aliases():
    from birdear.core import load_species_entries, name_key
    entries = load_species_entries()
    canon = [e[0] for e in entries]
    assert len(canon) == len(set(canon))
    assert ["Rock Dove", "Rock Pigeon"] == next(e for e in entries if e[0] == "Rock Dove")
    assert name_key("Paradise-flycatcher") == name_key("paradise Flycatcher")
