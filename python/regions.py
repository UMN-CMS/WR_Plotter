from .plotter import Region

def build_regions(era: str, category: str):
    if category == "dy_cr":
        return [
            Region("wr_mumu_resolved_dy_cr", "Muon",   True, f"$\\mu\\mu$\nResolved DY CR\n{era}"),
            Region("wr_ee_resolved_dy_cr",   "EGamma", True, f"ee\nResolved DY CR\n{era}"),
        ]
    elif category == "flavor_cr":
        return [
            Region("wr_resolved_flavor_cr", "Muon",   True, f"$e\\mu$\nResolved Flavor CR\n{era}"),
            Region("wr_resolved_flavor_cr", "EGamma", True, f"$e\\mu$\nResolved Flavor CR\n{era}"),
        ]
    raise ValueError(f"Unknown category: {category}")
