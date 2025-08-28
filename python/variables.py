from .plotter import Variable

def build_variables():
    defs = [
        ("mass_fourobject", r"$m_{lljj}$", "GeV"),
        ("pt_leading_jet", r"$p_{T}$ of the leading jet", "GeV"),
        ("mass_dijet", r"$m_{jj}$", "GeV"),
        ("pt_leading_lepton", r"$p_{T}$ of the leading lepton", "GeV"),
        ("eta_leading_lepton", r"$\eta$ of the leading lepton", ""),
        ("phi_leading_lepton", r"$\phi$ of the leading lepton", ""),
        ("pt_subleading_lepton", r"$p_{T}$ of the subleading lepton", "GeV"),
        ("eta_subleading_lepton", r"$\eta$ of the subleading lepton", ""),
        ("phi_subleading_lepton", r"$\phi$ of the subleading lepton", ""),
        ("eta_leading_jet", r"$\eta$ of the leading jet", ""),
        ("phi_leading_jet", r"$\phi$ of the leading jet", ""),
        ("pt_subleading_jet", r"$p_{T}$ of the subleading jet", "GeV"),
        ("eta_subleading_jet", r"$\eta$ of the subleading jet", ""),
        ("phi_subleading_jet", r"$\phi$ of the subleading jet", ""),
        ("mass_dilepton", r"$m_{ll}$", "GeV"),
        ("pt_dilepton", r"$p_{T}^{ll}$", "GeV"),
        ("pt_dijet", r"$p_{T}^{jj}$", "GeV"),
        ("mass_threeobject_leadlep", r"$m_{l_{\mathrm{pri}}jj}$", "GeV"),
        ("pt_threeobject_leadlep", r"$p^{T}_{l_{\mathrm{pri}}jj}$", "GeV"),
        ("mass_threeobject_subleadlep", r"$m_{l_{\mathrm{sec}}jj}$", "GeV"),
        ("pt_threeobject_subleadlep", r"$p^{T}_{l_{\mathrm{sec}}jj}$", "GeV"),
        ("pt_fourobject", r"$p^{T}_{lljj}$", "GeV"),
    ]
    return [Variable(n, t, u) for n, t, u in defs]
