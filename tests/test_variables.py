"""Tests for wrplotter.variables."""
from wrplotter.variables import Variable, build_variables


class TestVariable:

    def test_label_with_unit(self):
        v = Variable("pt", r"$p_T$", "GeV")
        assert v.label() == r"$p_T$ [GeV]"

    def test_label_without_unit(self):
        v = Variable("eta", r"$\eta$", "")
        assert v.label() == r"$\eta$"

    def test_frozen(self):
        v = Variable("x", "X", "GeV")
        try:
            v.name = "y"
            assert False, "should be frozen"
        except AttributeError:
            pass


class TestBuildVariables:

    def test_returns_list(self):
        assert isinstance(build_variables(), list)

    def test_nonempty(self):
        assert len(build_variables()) > 0

    def test_all_variable_instances(self):
        for v in build_variables():
            assert isinstance(v, Variable)

    def test_unique_names(self):
        names = [v.name for v in build_variables()]
        assert len(names) == len(set(names))

    def test_known_variables_present(self):
        names = {v.name for v in build_variables()}
        for expected in ["mass_fourobject", "pt_leading_jet", "mass_dilepton"]:
            assert expected in names, f"Expected variable '{expected}' not found"
