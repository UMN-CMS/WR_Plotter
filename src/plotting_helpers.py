import os
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Union
import numpy as np

def custom_log_formatter(y, pos):
    """
    Custom formatter for log-scaled y-axis.
    """
    if y == 1:
        return '1'
    elif y == 10:
        return '10'
    else:
        return f"$10^{{{int(np.log10(y))}}}$"


def set_y_label(ax, tot, variable):
    """
    Adjust the y-axis label based on bin width.
    """
    x_bins = tot.axes[0].edges
    bin_widths = np.round(np.diff(x_bins), 1)
    if np.all(bin_widths == bin_widths[0]):  # uniform bin width
        bin_width = bin_widths[0]
        formatted_bin_width = int(bin_width) if bin_width.is_integer() else f"{bin_width:.1f}"
        ax.set_ylabel(f"Events / {formatted_bin_width} {variable.unit}")
    else:
        ax.set_ylabel(f"Events / X {variable.unit}")

def _eos_mkdir(eos_dir: str) -> bool:
    try:
        subprocess.run(["xrdfs", "eosuser.cern.ch", "mkdir", "-p", eos_dir], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create directory on EOS: {e}")
        return False

def _save_and_upload(fig, tmp_suffix: str, fmt: str, eos_target: str,
                     dpi: int = None, jpg_quality: int = None):
    with tempfile.NamedTemporaryFile(suffix=tmp_suffix) as tmp_file:
        save_kwargs = {"format": fmt, "bbox_inches": "tight"}
        if dpi is not None and fmt != "pdf":
            save_kwargs["dpi"] = dpi
        if fmt == "jpeg" and jpg_quality is not None:
            try:
                save_kwargs["pil_kwargs"] = {"quality": jpg_quality, "optimize": True, "progressive": True}
            except TypeError:
                # Older Matplotlib: no pil_kwargs support
                pass

        fig.savefig(tmp_file.name, **save_kwargs)
        tmp_file.flush()

        while True:
            try:
                subprocess.run(
                    ["xrdcp", "-f", tmp_file.name, f"root://eosuser.cern.ch/{eos_target}"],
                    check=True, timeout=20
                )
                print(f"Uploaded: {eos_target}")
                break
            except subprocess.TimeoutExpired:
                print("xrdcp timed out. Retrying...")
            except subprocess.CalledProcessError as e:
                print(f"Failed to upload {eos_target}: {e}. Retrying in 5 seconds...")
                time.sleep(5)

def save_figure(fig, eos_path: Union[str, Path], dpi_jpg: int = 600, jpg_quality: int = 95):
    """
    Save `fig` to EOS as both PDF (vector) and JPEG (raster, dpi_jpg).
    `eos_path` can include an extension (.pdf/.jpg) or not; we normalize to a base.
    """
    eos_path = str(eos_path)
    base, ext = os.path.splitext(eos_path)
    eos_base = base if ext.lower() in (".pdf", ".jpg", ".jpeg", ".png") else eos_path

    eos_dir = os.path.dirname(eos_base)
    print(eos_dir)
    if not _eos_mkdir(eos_dir):
        return

    # PDF (vector)
    _save_and_upload(fig, ".pdf", "pdf", eos_base + ".pdf")

    # JPEG (raster @ 300 dpi by default)
    _save_and_upload(fig, ".jpg", "jpeg", eos_base + ".jpg", dpi=dpi_jpg, jpg_quality=jpg_quality)
