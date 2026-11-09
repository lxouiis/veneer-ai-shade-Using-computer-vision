import shutil
from IPython.display import FileLink

# --------------------------------------------------------------------------
# Zips evaluate.py's worst/best comparison PNGs for download. Run this in
# the same Kaggle session, after evaluate.py has finished and populated
# OUTPUT_DIR.
# --------------------------------------------------------------------------
OUTPUT_DIR = "/kaggle/working/outputs_eval_combined"
ZIP_BASENAME = "/kaggle/working/outputs_eval_combined"  # shutil appends .zip itself

zip_path = shutil.make_archive(ZIP_BASENAME, "zip", OUTPUT_DIR)
print(f"Zipped to: {zip_path}")

# Clicking this link downloads the zip directly from the notebook - no need
# to commit/save a version first, unlike the Output tab's file browser.
FileLink(zip_path)
