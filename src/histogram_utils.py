def rebin_histogram(hist, n_rebin):
    """
    Rebin the given histogram using a specified factor.
    This is a generic function that assumes the histogram supports slicing.
    """
    try:
        # Your rebin logic (here, a placeholder using slicing)
        return hist[::(1j * n_rebin)]
    except Exception as e:
        raise RuntimeError(f"Error during rebinning: {e}")

def scale_histogram(hist, scale_factor):
    """
    Scale the histogram data by a given factor.
    """
    try:
        # Assuming hist supports element-wise multiplication.
        return hist * scale_factor
    except Exception as e:
        raise RuntimeError(f"Error during scaling: {e}")
