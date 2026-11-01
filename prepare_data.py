"""
Builds the small dataset used by the website's practical demonstration.

You do NOT need to run this script: its output, data/network_sample.csv, is
already included in the project. It is provided so the sample can be checked
and reproduced.

Source dataset: NSL-KDD (Tavallaee et al., 2009), a public, labelled network
intrusion detection dataset distributed by the Canadian Institute for
Cybersecurity, University of New Brunswick:
    https://www.unb.ca/cic/datasets/nsl.html
The file is downloaded from a public GitHub mirror of the original release.

Sampling: 1,900 normal connections + 100 attack connections, drawn at random
from KDDTrain+. Attacks are kept rare on purpose (5%) because Isolation Forest
assumes anomalies are "few and different". The full file has about 47% attacks.

Run:  python prepare_data.py
"""

from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
OUT = Path(__file__).parent / "data" / "network_sample.csv"
N_NORMAL, N_ATTACK, SEED = 1900, 100, 1

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]


def main() -> None:
    df = pd.read_csv(URL, header=None, names=COLUMNS)
    normal = df[df["label"] == "normal"].sample(N_NORMAL, random_state=SEED)
    attack = df[df["label"] != "normal"].sample(N_ATTACK, random_state=SEED)
    sample = pd.concat([normal, attack]).sample(frac=1, random_state=SEED)
    sample = sample.drop(columns="difficulty").reset_index(drop=True)
    OUT.parent.mkdir(exist_ok=True)
    sample.to_csv(OUT, index=False)
    print(f"Saved {len(sample):,} records to {OUT}")
    print(sample["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
