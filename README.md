# Isolation Forest — Algorithms Course Project

A simple educational website that explains the **Isolation Forest** algorithm and ends with a small
practical demonstration on real network traffic data.

The page is one scrollable presentation with six sections:

1. **Definition** — what Isolation Forest and anomaly detection are, and why the algorithm is unsupervised
2. **How the algorithm works** — the steps, an illustration, and the key terms (isolation tree, random
   splitting, path length, average path length, anomaly score)
3. **Applications** — six use cases with examples
4. **Pseudocode** — with a short explanation
5. **Time complexity** — training O(T × ψ × log ψ), prediction O(N × T × log ψ)
6. **Practical demonstration** — runs scikit-learn's Isolation Forest on 2,000 real network connections

## Files

```
isolation-forest-site/
├── app.py                    # The whole website
├── prepare_data.py           # (Optional) re-creates the demo dataset
├── data/network_sample.csv   # Demo dataset (included, loads automatically)
├── requirements.txt          # Python libraries
├── .streamlit/config.toml    # Light theme
└── README.md
```

## The demonstration dataset

`data/network_sample.csv` is a random sample of **real** network connections from **NSL-KDD**, a public,
labelled intrusion detection dataset distributed by the Canadian Institute for Cybersecurity (University of
New Brunswick): <https://www.unb.ca/cic/datasets/nsl.html>.

- 1,900 normal connections + 100 attack connections (5%), drawn at random from `KDDTrain+`.
- Attacks are kept rare on purpose, because Isolation Forest assumes anomalies are few.
- The model uses only the numeric features. The `label` column is never given to the model; it is used
  only after prediction to explain the results.
- `prepare_data.py` shows exactly how the sample was made (downloaded from a public GitHub mirror of the
  original files). You do not need to run it.

NSL-KDD was used instead of CICIDS2017 because CICIDS2017 is only distributed as very large files from the
university server and cannot be downloaded automatically in a reliable way.

## Run locally

Requirements: Python 3.10 or newer.

```bash
# 1. Open a terminal inside the project folder (the one that contains app.py)

# 2. Install the libraries (only the first time)
python -m pip install -r requirements.txt

# 3. Start the website
python -m streamlit run app.py
```

The website opens in your browser at <http://localhost:8501>. Keep the terminal open while using it;
press `Ctrl + C` in the terminal to stop it.

On macOS, use `python3` instead of `python`. If `python` is not found on Windows, try `py`.

## Publish on Streamlit Community Cloud

1. **Create a GitHub account** at <https://github.com> (if you do not have one).
2. **Create a new public repository**: click **+** → **New repository**, name it (for example
   `isolation-forest`), choose **Public**, and click **Create repository**.
3. **Upload the files**: click **uploading an existing file**, then drag *everything inside* the project
   folder (including the `data` and `.streamlit` folders) into the page, and click **Commit changes**.
4. **Deploy**: go to <https://share.streamlit.io>, sign in with GitHub, click **Create app** → **Deploy a
   public app from GitHub**, choose your repository, branch `main`, main file `app.py`, then **Deploy**.
5. After a few minutes you get a public link (for example `https://your-name-isolation-forest.streamlit.app`)
   that anyone can open.

If the app has not been used for a while it goes to sleep; the visitor clicks **"Yes, get this app back
up!"** and waits about a minute.

## References

- F. T. Liu, K. M. Ting, Z.-H. Zhou. "Isolation Forest." *IEEE International Conference on Data Mining*, 2008.
- M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani. "A Detailed Analysis of the KDD CUP 99 Data Set."
  *IEEE CISDA*, 2009.
