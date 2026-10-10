"""Tests for wrplotter.systematics: which variations exist, the nominal fallback,
envelopes and the band."""
import numpy as np
import pytest
import uproot
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter.histogram_cache import read_hist
from wrplotter.systematics import (
    band_error2, clear_cache, sample_variations, sources_in_files, total_variations, variation_key,
)

REGION, VAR = "wr_ee_resolved_sr", "mass_fourobject"


@pytest.fixture(autouse=True)
def _fresh():
    clear_cache()
    yield
    clear_cache()


def _h(values):
    h = Hist(Regular(len(values), 0, 100, name="x"), storage=Weight())
    h.view().value[...] = values
    h.view().variance[...] = values
    return h


def _write(path, variations, nominal=(10.0, 20.0)):
    """{(source, direction, region): values} plus the nominal of REGION."""
    with uproot.recreate(path) as f:
        f[f"{REGION}/{VAR}_{REGION}"] = _h(list(nominal))
        for (source, direction, region), values in variations.items():
            f[variation_key(source, direction, region, VAR)] = _h(values)
    return path


def _variations(paths, systematics):
    """sample_variations of a sample with one file per era in *paths* (read without rebinning)."""
    nominals = {p: read_hist(p, f"{REGION}/{VAR}_{REGION}") for p in paths}
    return sample_variations(nominals, REGION, VAR, systematics, load=read_hist)


def test_sources_in_files(tmp_path):
    path = _write(tmp_path / "a.root", {
        ("jes", "up", REGION): [11, 21], ("jes", "down", REGION): [9, 19],
        ("scalemurf", "up", REGION): [12, 22], ("musmear", "down", "wr_mumu_resolved_sr"): [1, 1],
    })
    assert sources_in_files([path]) == {"jes", "scalemurf", "musmear"}


def test_nominal_only_file_gives_no_band(tmp_path):
    path = _write(tmp_path / "a.root", {})
    assert _variations([path], ["jes", "scale"]) == {}


def test_source_missing_in_region_uses_nominal(tmp_path):
    # jes was produced, but only in another region: it does not affect REGION.
    path = _write(tmp_path / "a.root", {("jes", "up", "wr_mumu_resolved_sr"): [1, 1],
                                        ("jes", "down", "wr_mumu_resolved_sr"): [1, 1]})
    out = _variations([path], ["jes", "pileup"])
    assert set(out) == {"jes"}
    for direction in ("up", "down"):
        np.testing.assert_array_equal(out["jes"][direction].values(), [10.0, 20.0])


def test_scale_envelope(tmp_path):
    path = _write(tmp_path / "a.root", {
        ("scalemur", "up", REGION): [12, 18], ("scalemur", "down", REGION): [9, 21],
        ("scalemuf", "up", REGION): [11, 23], ("scalemuf", "down", REGION): [8, 20],
        # scalemurf not produced for this sample: envelope of the other two
    })
    out = _variations([path], ["scale"])
    np.testing.assert_array_equal(out["scale"]["up"].values(), [12, 23])
    np.testing.assert_array_equal(out["scale"]["down"].values(), [8, 18])


# ---- several eras (one file each) of one sample ----

JES = {("jes", "up", REGION): [11, 22], ("jes", "down", REGION): [9, 18]}


def test_era_without_variations_adds_its_nominal(tmp_path, caplog):
    # Only the first era was produced with systematics.  The second must add its
    # nominal to each variation, not nothing: that made the band as large as the
    # second era's whole yield.
    a = _write(tmp_path / "a.root", JES)
    b = _write(tmp_path / "b.root", {}, nominal=(100.0, 200.0))
    with caplog.at_level("WARNING"):
        out = _variations([a, b], ["jes"])
    np.testing.assert_array_equal(out["jes"]["up"].values(), [111, 222])
    np.testing.assert_array_equal(out["jes"]["down"].values(), [109, 218])
    assert f"{b} has no systematic variations" in caplog.text


def test_era_missing_one_source_adds_its_nominal(tmp_path, caplog):
    a = _write(tmp_path / "a.root", {**JES, ("pileup", "up", REGION): [12, 21],
                                     ("pileup", "down", REGION): [8, 19]})
    b = _write(tmp_path / "b.root", {("jes", "up", REGION): [101, 202],
                                     ("jes", "down", REGION): [99, 198]}, nominal=(100.0, 200.0))
    with caplog.at_level("WARNING"):
        out = _variations([a, b], ["jes", "pileup"])
    np.testing.assert_array_equal(out["jes"]["up"].values(), [112, 224])
    np.testing.assert_array_equal(out["pileup"]["up"].values(), [112, 221])
    assert f"{b} has no pileup variations" in caplog.text


def test_source_not_affecting_the_region_in_one_era_is_quiet(tmp_path, caplog):
    # b has jes, but not in REGION: there jes does not affect REGION, so its nominal is used.
    a = _write(tmp_path / "a.root", JES)
    b = _write(tmp_path / "b.root", {("jes", "up", "wr_mumu_resolved_sr"): [1, 1],
                                     ("jes", "down", "wr_mumu_resolved_sr"): [1, 1]},
               nominal=(100.0, 200.0))
    with caplog.at_level("WARNING"):
        out = _variations([a, b], ["jes"])
    np.testing.assert_array_equal(out["jes"]["up"].values(), [111, 222])
    assert caplog.text == ""


def test_scale_envelope_of_the_era_sums(tmp_path):
    a = _write(tmp_path / "a.root", {("scalemur", "up", REGION): [12, 18],
                                     ("scalemur", "down", REGION): [9, 21]})
    b = _write(tmp_path / "b.root", {("scalemur", "up", REGION): [101, 199],
                                     ("scalemur", "down", REGION): [99, 201]}, nominal=(100.0, 200.0))
    out = _variations([a, b], ["scale"])
    # era sums: up [113, 217], down [108, 222]
    np.testing.assert_array_equal(out["scale"]["up"].values(), [113, 222])
    np.testing.assert_array_equal(out["scale"]["down"].values(), [108, 217])


def test_unreadable_file_raises(tmp_path):
    # It used to count as a file without variations, turning the band statistical without notice.
    bad = tmp_path / "bad.root"
    bad.write_bytes(b"not a ROOT file")
    with pytest.raises((ValueError, OSError)):
        sources_in_files([bad])


def test_total_variations_fills_in_nominal():
    a_nom, b_nom = _h([10.0, 20.0]), _h([1.0, 2.0])
    a_var = {"jes": {"up": _h([12.0, 22.0]), "down": _h([8.0, 18.0])}}
    total = total_variations([(a_nom, a_var), (b_nom, {})])
    np.testing.assert_array_equal(total["jes"]["up"].values(), [13, 24])
    np.testing.assert_array_equal(total["jes"]["down"].values(), [9, 20])


def test_band_error2():
    nominal = np.array([10.0, 20.0])
    variations = {"jes": {"up": _h([12.0, 20.0]), "down": _h([9.0, 17.0])},
                  "pileup": {"up": _h([11.0, 20.0]), "down": _h([10.0, 20.0])}}
    err2, has_syst = band_error2(nominal, variations)
    np.testing.assert_allclose(err2, [2.0 ** 2 + 1.0 ** 2, 3.0 ** 2])
    assert has_syst


def test_band_error2_identical_is_not_a_syst():
    nominal = np.array([10.0, 20.0])
    err2, has_syst = band_error2(nominal, {"musmear": {"up": _h([10.0, 20.0]),
                                                       "down": _h([10.0, 20.0])}})
    assert not has_syst
    np.testing.assert_array_equal(err2, 0)
    assert band_error2(nominal, None) == (pytest.approx([0, 0]), False)
