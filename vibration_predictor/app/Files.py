from pathlib import Path
import pandas as pd
import numpy as np


def list_ims_files(folder):
    files = list(Path(folder).glob("*"))
    files = [f for f in files if f.is_file()]
    files = sorted(files, key=lambda x: x.stat().st_mtime)
    return files


def load_ims_file(path):
    df = pd.read_csv(
        path,
        sep="\t",  # IMS uses tab-separated columns
        header=None,  # no header rows
        dtype=np.float32  # reduces memory consumption
    )

    return df.values
