import numpy as np
import tempfile
import os 
import subprocess
import time

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

def save_figure(fig, eos_path):
    # Ensure EOS directory exists
    eos_dir = os.path.dirname(eos_path)
    try:
        subprocess.run(["xrdfs", "eosuser.cern.ch", "mkdir", "-p", eos_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create directory on EOS: {e}")
        return  # Exit function if directory creation fails

    # Save plot to a temporary PDF file and upload it to EOS
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
        fig.savefig(tmp_file.name, format="pdf")  # Save the plot as a PDF
        tmp_file.flush()  # Ensure all data is written to disk

        # Upload the temporary PDF file to EOS
        while True:
            try:
                subprocess.run(
                    ["xrdcp", "-f", tmp_file.name, f"root://eosuser.cern.ch/{eos_path}"], 
                    check=True,
                    timeout=5
                )
                print(f"File uploaded: {eos_path}.")
                break  # Exit the loop on successful upload
            except subprocess.TimeoutExpired:
                print("xrdcp command timed out. Retrying...")
            except subprocess.CalledProcessError as e:
                print(f"Failed to upload file to EOS: {e}. Retrying in 5 seconds...")
            time.sleep(1)
