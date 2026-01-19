# python/variables.py
from __future__ import annotations
from dataclasses import dataclass, replace
import logging
import numpy as np
import hist

@dataclass(frozen=True)
class Variable:
    name: str
    tlatex_alias: str
    unit: str = ""

    # default: regular binning (uniform)
    nbins: int | None = None
    xmin: float | None = None
    xmax: float | None = None

    def label(self) -> str:
        return f"{self.tlatex_alias} [{self.unit}]" if self.unit else self.tlatex_alias

#    def axis(self, override_bins: list[float] | None = None) -> hist.axis.Axis:
#        """
#        Return a boost-histogram axis. If override_bins is given, use variable-width bins.
#        Otherwise fall back to the variable's regular (uniform) binning.
#        """
#        if override_bins is not None:
#            edges = np.asarray(override_bins, dtype=float)
#            if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
#                raise ValueError(f"{self.name}: invalid override_bins")
#            return hist.axis.Variable(edges, name=self.name, label=self.label())
#
#        if None not in (self.nbins, self.xmin, self.xmax):
#            return hist.axis.Regular(self.nbins, self.xmin, self.xmax, name=self.name, label=self.label())
#
#        raise ValueError(f"Variable {self.name} has no binning (nbins/xmin/xmax) and no override provided")

    def print(self) -> None:
        logging.info(
            f"Variable(name={self.name}, tlatex={self.tlatex_alias}, unit={self.unit}, "
            f"nbins={self.nbins}, range=({self.xmin},{self.xmax}))"
        )

def build_variables() -> list[Variable]:
    # Give each variable sensible uniform defaults; tune as you like
    defs = [
        # Resolved region variables
        Variable("mass_fourobject", r"$m_{lljj}$", "GeV", nbins=40, xmin=0,   xmax=4000),
        Variable("pt_leading_jet", r"$p_{T}$ of the leading jet", "GeV", nbins=30, xmin=0, xmax=600),
        Variable("mass_dijet", r"$m_{jj}$", "GeV", nbins=36, xmin=0,   xmax=3600),
        Variable("pt_leading_lepton", r"$p_{T}$ of the leading lepton", "GeV", nbins=30, xmin=0, xmax=300),
        Variable("eta_leading_lepton", r"$\eta$ of the leading lepton", "", nbins=50, xmin=-2.5, xmax=2.5),
        Variable("phi_leading_lepton", r"$\phi$ of the leading lepton", "", nbins=64, xmin=-3.2, xmax=3.2),
        Variable("pt_subleading_lepton", r"$p_{T}$ of the subleading lepton", "GeV", nbins=30, xmin=0, xmax=300),
        Variable("eta_subleading_lepton", r"$\eta$ of the subleading lepton", "", nbins=50, xmin=-2.5, xmax=2.5),
        Variable("phi_subleading_lepton", r"$\phi$ of the subleading lepton", "", nbins=64, xmin=-3.2, xmax=3.2),
        Variable("eta_leading_jet", r"$\eta$ of the leading jet", "", nbins=60, xmin=-3.0, xmax=3.0),
        Variable("phi_leading_jet", r"$\phi$ of the leading jet", "", nbins=64, xmin=-3.2, xmax=3.2),
        Variable("pt_subleading_jet", r"$p_{T}$ of the subleading jet", "GeV", nbins=30, xmin=0, xmax=400),
        Variable("eta_subleading_jet", r"$\eta$ of the subleading jet", "", nbins=60, xmin=-3.0, xmax=3.0),
        Variable("phi_subleading_jet", r"$\phi$ of the subleading jet", "", nbins=64, xmin=-3.2, xmax=3.2),
        Variable("mass_dilepton", r"$m_{ll}$", "GeV", nbins=60, xmin=0, xmax=300),
        Variable("pt_dilepton", r"$p_{T}^{ll}$", "GeV", nbins=40, xmin=0, xmax=400),
        Variable("pt_dijet", r"$p_{T}^{jj}$", "GeV", nbins=40, xmin=0, xmax=800),
        Variable("mass_threeobject_leadlep", r"$m_{l_{\mathrm{pri}}jj}$", "GeV", nbins=40, xmin=0, xmax=4000),
        Variable("pt_threeobject_leadlep", r"$p^{T}_{l_{\mathrm{pri}}jj}$", "GeV", nbins=40, xmin=0, xmax=800),
        Variable("mass_threeobject_subleadlep", r"$m_{l_{\mathrm{sec}}jj}$", "GeV", nbins=40, xmin=0, xmax=4000),
        Variable("pt_threeobject_subleadlep", r"$p^{T}_{l_{\mathrm{sec}}jj}$", "GeV", nbins=40, xmin=0, xmax=800),
        Variable("pt_fourobject", r"$p^{T}_{lljj}$", "GeV", nbins=50, xmin=0, xmax=1500),
        Variable("LSF_leading_AK8Jets", r"LSF3_leadingAK8Jets", "" , nbins = 100, xmin=0, xmax=1.1),
        Variable("pt_leading_loose_lepton",r"$p_{T}$ of the leading loose lepton","GeV",nbins=30, xmin=0, xmax=300),
        Variable("eta_leading_loose_lepton", r"$\eta$ of the leading loose lepton", "", nbins=50, xmin=-2.5, xmax=2.5),
        Variable("phi_leading_loose_lepton", r"$\phi$ of the leading loose lepton", "", nbins=64, xmin=-3.2, xmax=3.2),
        Variable("pt_leading_AK8Jets", r"$p_{T}$ of the leading AK8 jet", "GeV", nbins=30, xmin=0, xmax=600),
        Variable("eta_leading_AK8Jets", r"$\eta$ of the leading AK8 jet", "", nbins=60, xmin=-3.0, xmax=3.0),
        Variable("phi_leading_AK8Jets", r"$\phi$ of the leading AK8 jet", "", nbins=64, xmin=-3.2, xmax=3.2),
        Variable("pt_twoobject", r"$p^{T}_{lj}$", "GeV", nbins=50, xmin=0, xmax=1500),
        Variable("mass_twoobject", r"$m_{lj}$", "GeV", nbins=40, xmin=0,   xmax=4000),
    ]
    return defs
