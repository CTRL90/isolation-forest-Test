"""
Isolation Forest — an educational website for an Algorithms course project.

Run locally:   streamlit run app.py

The page is organised as an academic presentation:
    1. Definition   2. How it works   3. Applications
    4. Pseudocode   5. Time complexity   6. Practical demonstration
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

DATA_FILE = Path(__file__).parent / "data" / "network_sample.csv"
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b8bec9", "#1f2937", "#6b7280"
CATEGORICAL = ["protocol_type", "service", "flag"]   # text columns, not used by the model
LABEL = "label"                                         # ground truth, never used by the model

st.set_page_config(page_title="Isolation Forest — Algorithms Project", page_icon="🌲",
                   layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: #111827; }
h2 { border-bottom: 1px solid #e5e7eb; padding-bottom: .35rem; margin-top: 2.2rem !important; }
.block-container { max-width: 860px; padding-top: 2.5rem; }
.subtitle { color: #4b5563; font-size: 1.08rem; margin-top: -.6rem; }
.key-idea { border-left: 4px solid #2a78d6; background: #f3f7fd; padding: .8rem 1.1rem;
            border-radius: 0 8px 8px 0; margin: 1rem 0; color: #1f2937; }
.caution  { border-left: 4px solid #d99a00; background: #fdf8ec; padding: .8rem 1.1rem;
            border-radius: 0 8px 8px 0; margin: 1rem 0; color: #1f2937; }
.pseudo { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem 1.3rem;
          font-family: 'JetBrains Mono', monospace; font-size: .88rem; line-height: 1.6;
          white-space: pre-wrap; color: #111827; }
.pseudo b { color: #1d4ed8; }
.toc a { color: #1d4ed8; text-decoration: none; margin-right: 1.1rem; font-weight: 500;
         display: inline-block; white-space: nowrap; }
.toc { padding: .6rem 0 .2rem 0; border-top: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def c(n: int) -> float:
    """c(n): average path length of an unsuccessful search in a binary search tree."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


def toy_points() -> np.ndarray:
    """60 clustered 2-D points plus one far-away point (the last row)."""
    rng = np.random.default_rng(7)
    cluster = rng.normal([5, 5], 0.9, (60, 2))
    return np.vstack([cluster, [[9.2, 1.4]]])


def isolate(points: np.ndarray, target: int, seed: int) -> list:
    """Split randomly (random feature, random value) until `target` is alone.

    Returns the splits as (feature, value, region) so they can be drawn.
    This is exactly what one isolation tree does along the path of one record.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(points))
    region = [points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()]
    splits = []
    while len(idx) > 1:
        f = int(rng.integers(0, 2))
        lo, hi = points[idx, f].min(), points[idx, f].max()
        if lo == hi:
            continue
        v = float(rng.uniform(lo, hi))
        splits.append((f, v, tuple(region)))
        if points[target, f] < v:
            idx = idx[points[idx, f] < v]
            region[1 if f == 0 else 3] = v
        else:
            idx = idx[points[idx, f] >= v]
            region[0 if f == 0 else 2] = v
    return splits


def isolation_figure(points, target, splits, color, title) -> go.Figure:
    fig = go.Figure()
    others = np.delete(np.arange(len(points)), target)
    fig.add_trace(go.Scatter(x=points[others, 0], y=points[others, 1], mode="markers",
                             marker=dict(color=GREY, size=7), hoverinfo="skip", showlegend=False))
    for i, (f, v, (x0, x1, y0, y1)) in enumerate(splits, 1):
        if f == 0:
            fig.add_shape(type="line", x0=v, x1=v, y0=y0, y1=y1, line=dict(color=color, width=1.6))
        else:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=v, y1=v, line=dict(color=color, width=1.6))
    fig.add_trace(go.Scatter(x=[points[target, 0]], y=[points[target, 1]], mode="markers",
                             marker=dict(color=color, size=15, line=dict(color="white", width=2)),
                             hoverinfo="skip", showlegend=False))
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    y0, y1 = points[:, 1].min(), points[:, 1].max()
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="#d1d5db", width=1))
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=INK)), height=360,
                      margin=dict(l=10, r=10, t=45, b=10), plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(visible=False, range=[x0 - 0.4, x1 + 0.4])
    fig.update_yaxes(visible=False, range=[y0 - 0.4, y1 + 0.4])
    return fig


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def run_isolation_forest(df: pd.DataFrame):
    """Train Isolation Forest on the numeric features only (no labels)."""
    features = [col for col in df.columns
                if col not in CATEGORICAL + [LABEL] and df[col].nunique() > 1]
    X = df[features]
    model = IsolationForest(n_estimators=100, max_samples=256, contamination="auto", random_state=42)
    model.fit(X)
    result = df.copy()
    result["anomaly_score"] = -model.score_samples(X)            # s(x, ψ), between 0 and 1
    result["avg_path_length"] = -np.log2(result["anomaly_score"]) * c(int(model.max_samples_))
    result["prediction"] = np.where(model.predict(X) == -1, "Anomaly", "Normal")
    return result, features, int(model.max_samples_)


# --------------------------------------------------------------------------- #
# Title
# --------------------------------------------------------------------------- #

st.title("Isolation Forest")
st.markdown('<p class="subtitle">An anomaly detection algorithm that finds unusual records '
            'by isolating them with random splits.</p>', unsafe_allow_html=True)
st.markdown('<div class="toc">'
            '<a href="#definition">1. Definition</a><a href="#how-it-works">2. How it works</a>'
            '<a href="#applications">3. Applications</a><a href="#pseudocode">4. Pseudocode</a>'
            '<a href="#time-complexity">5. Time complexity</a><a href="#demonstration">6. Demonstration</a>'
            '</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 1. Definition
# --------------------------------------------------------------------------- #

st.header("1. Definition", anchor="definition")
st.markdown("""
**Isolation Forest** is a machine learning algorithm for **anomaly detection**, introduced by Liu, Ting and
Zhou in 2008. It builds many random binary trees and measures how quickly each record can be separated
("isolated") from the rest of the data.

**Anomaly detection** means finding records that do not follow the pattern of the majority. For example,
among thousands of ordinary network connections, one connection that sends an unusually large amount of
data may be an anomaly.

**Why it is unsupervised.** The algorithm does not need labels such as "normal" or "attack". It learns
only from the data values themselves. This is important because labelled data is often expensive or
impossible to obtain.

**How it differs from classification.** A classification algorithm (for example, a decision tree
classifier) learns from labelled examples and then assigns each new record to a known class. Isolation
Forest has no classes to learn. It answers a different question: *how unusual is this record compared
with the others?*

**Why it is useful in cybersecurity.** New attacks often have no labelled examples yet. Because Isolation
Forest looks for unusual behaviour instead of known attack patterns, it can draw attention to suspicious
activity that a classifier trained on past attacks might miss.
""")

# --------------------------------------------------------------------------- #
# 2. How it works
# --------------------------------------------------------------------------- #

st.header("2. How the Algorithm Works", anchor="how-it-works")
st.markdown('<div class="key-idea"><b>Main idea:</b> anomalies are <b>few</b> and <b>different</b>, '
            'so they are easier to isolate. They need <b>fewer random splits</b>, which means a '
            '<b>shorter path length</b> in the tree, than normal records.</div>', unsafe_allow_html=True)

st.markdown("""
**Step by step**

1. **Randomly select a subset** of ψ records from the dataset (by default ψ = 256).
2. **Randomly select a feature**, for example *bytes sent*.
3. **Choose a random split value** between that feature's minimum and maximum in the current group.
4. **Split the records into two groups**: values below the split go left, the others go right.
5. **Repeat the splitting** inside each group until every record is alone (or a depth limit is reached).
   The result is one **isolation tree**. Steps 1–5 are repeated to build T trees (by default T = 100).
6. **Calculate the path length** of each record: the number of splits needed to isolate it in a tree.
7. **Calculate the average path length** of each record over all T trees.
8. **Calculate the anomaly score** from the average path length (formula below).
9. **Identify records with high anomaly scores** (above 0.5 in the original paper) as anomalies.
""")

st.markdown("**Illustration.** The same random splitting procedure applied to two different points "
            "(each line is one random split):")
pts = toy_points()
cluster = pts[:-1]
central = int(np.argmin(((cluster - cluster.mean(axis=0)) ** 2).sum(axis=1)))
outlier = len(pts) - 1
split_normal = isolate(pts, central, seed=10)
split_outlier = isolate(pts, outlier, seed=1002)
left, right = st.columns(2)
left.plotly_chart(isolation_figure(pts, central, split_normal, BLUE,
                  f"Normal record: {len(split_normal)} splits"), width="stretch",
                  config={"displayModeBar": False})
right.plotly_chart(isolation_figure(pts, outlier, split_outlier, ORANGE,
                   f"Unusual record: {len(split_outlier)} splits"), width="stretch",
                   config={"displayModeBar": False})
st.caption("The grey frame is the full range of both features, where splitting starts. "
           "The record inside the dense group (blue) needs many splits before it is alone. The far-away "
           "record (orange) is separated after only a couple of splits. Over 1,000 random trees on this "
           "toy data, the averages are about 10.7 and 2.1 splits.")

st.markdown("**Key terms**")
st.markdown("""
| Term | Simple meaning |
|---|---|
| **Isolation Tree** | A binary tree built by splitting a random sample of records with random features and random split values. |
| **Random Splitting** | Choosing a feature at random and a split value at random within its range. No learning or optimisation is involved. |
| **Path Length** h(x) | The number of edges from the root to the leaf where record *x* ends up, i.e. how many splits isolated it. |
| **Average Path Length** E[h(x)] | The path length of *x* averaged over all trees. Averaging removes the luck of a single random tree. |
| **Anomaly Score** s(x, ψ) | A number between 0 and 1 computed from the average path length. Close to 1 means very likely an anomaly. |
""")

st.markdown("The anomaly score compares a record's average path length with **c(ψ)**, the typical path "
            "length in a tree built from ψ records:")
st.latex(r"s(x,\psi) = 2^{-\frac{E[h(x)]}{c(\psi)}}, \qquad "
         r"c(\psi) = 2H(\psi-1) - \frac{2(\psi-1)}{\psi}")
st.markdown("""
where H(i) ≈ ln(i) + 0.5772 is the harmonic number. Reading the score:
- **s close to 1**: the path is much shorter than usual → **anomaly**
- **s ≈ 0.5**: the path has a typical length → no clear sign of anomaly
- **s much smaller than 0.5**: the record is hard to isolate → **normal**
""")

# --------------------------------------------------------------------------- #
# 3. Applications
# --------------------------------------------------------------------------- #

st.header("3. Applications and Use Cases", anchor="applications")
st.markdown("""
| Application | How Isolation Forest helps | Simple example |
|---|---|---|
| **Network traffic anomaly detection** | Flags connections whose features (bytes, duration, error rates) differ from usual traffic. | A host suddenly opens hundreds of connections to different ports in one second. |
| **Fraud detection** | Highlights unusual activity on accounts or claims for review. | An insurance claim with an amount and timing unlike all similar claims. |
| **Unusual financial transactions** | Scores card or bank transactions by how unusual they are. | A card used for a large purchase in another country minutes after a local purchase. |
| **System monitoring** | Detects abnormal combinations of CPU, memory and response time on servers. | A server with normal CPU but response times ten times higher than usual. |
| **Abnormal sensor readings** | Finds faulty or unusual measurements in industrial or IoT sensors. | A temperature sensor reporting 300 °C in a room where others report 22 °C. |
| **Unusual user behaviour** | Spots accounts that behave differently from their peers. | An employee account downloading thousands of files at 3 a.m. |
""")
st.markdown('<div class="caution"><b>Anomaly detection ≠ confirmed attack classification.</b> '
            'Isolation Forest only says that a record is <i>unusual</i>. An unusual record may be an '
            'attack, but it may also be a rare legitimate event (for example, a large backup transfer). '
            'Flagged records are a starting point for investigation, not proof of an attack.</div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 4. Pseudocode
# --------------------------------------------------------------------------- #

st.header("4. Pseudocode", anchor="pseudocode")
PSEUDO_STYLE = ("background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:1rem 1.3rem;"
                "font-family:'JetBrains Mono',Consolas,monospace;font-size:.88rem;line-height:1.6;"
                "white-space:pre-wrap;color:#111827;margin:0")
PSEUDOCODE = """<b style="color:#1d4ed8">Algorithm:</b> Isolation Forest

<b style="color:#1d4ed8">Input:</b>
- Dataset X
- Number of trees T
- Subsample size ψ

<b style="color:#1d4ed8">Output:</b>
- Anomaly predictions

1. Initialize an empty collection of trees.
2. <b style="color:#1d4ed8">Repeat</b> T times:
   a. Randomly select ψ records from X.
   b. Build an Isolation Tree using random features and random split values.
   c. Add the tree to the collection.
3. <b style="color:#1d4ed8">For each</b> record x:
   a. Calculate its path length in every tree.
   b. Calculate the average path length.
   c. Calculate the anomaly score.
   d. Classify the record as normal or anomalous.
4. <b style="color:#1d4ed8">Return</b> the predictions."""
st.html(f'<pre style="{PSEUDO_STYLE}">{PSEUDOCODE}</pre>')
st.markdown("""
**Explanation.** Step 2 is the **training phase**: each tree is built from a small random sample, not
from the whole dataset, which keeps training fast and helps separate anomalies. Inside each tree, a
node is split using a random feature and a random value until records are isolated (in practice, tree
height is limited to about log₂ ψ). Step 3 is the **scoring phase**: every record is passed down every
tree, its path lengths are averaged, and the average is converted into the score s(x, ψ). In step 3d a
record is classified as anomalous when its score is above a threshold (0.5 in the original paper).
""")

# --------------------------------------------------------------------------- #
# 5. Time complexity
# --------------------------------------------------------------------------- #

st.header("5. Time Complexity", anchor="time-complexity")
a, b = st.columns(2)
with a:
    st.markdown("**Training**")
    st.latex(r"O(T \times \psi \times \log \psi)")
with b:
    st.markdown("**Prediction**")
    st.latex(r"O(N \times T \times \log \psi)")
st.markdown("""
- **T**: number of trees
- **ψ**: subsample size (records used to build each tree)
- **N**: number of records being analysed

**Training.** Each tree is built from only ψ records. At each level of the tree, every record is compared
with one split value, so a level costs about ψ operations. A tree built with random splits has an
expected height of about log ψ levels, so one tree costs ψ × log ψ, and T trees cost **T × ψ × log ψ**.
Notice that N does not appear: training cost does not grow with the size of the dataset.

**Prediction.** To score one record, it travels from the root to a leaf in each of the T trees, which takes
about log ψ steps per tree. For N records this gives **N × T × log ψ**.

These are **expected** complexity estimates under typical assumptions (random splits produce reasonably
balanced trees). They are **not guaranteed worst-case bounds**: an unlucky sequence of splits can create a
deeper tree.
""")

# --------------------------------------------------------------------------- #
# 6. Practical demonstration
# --------------------------------------------------------------------------- #

st.header("6. Practical Demonstration", anchor="demonstration")
df = load_data()
n_attack = int((df[LABEL] != "normal").sum())
st.markdown(f"""
**Dataset.** A sample of **{len(df):,} real network connections** from **NSL-KDD**, a public, labelled
network intrusion dataset distributed by the Canadian Institute for Cybersecurity (University of New
Brunswick). The sample contains {len(df) - n_attack:,} normal connections and {n_attack} attack
connections, chosen at random. Attacks were kept rare (5%) because Isolation Forest assumes anomalies
are few. The file is included with the website and loads automatically.
""")

st.markdown("**Step 1 — Preview of the records** (first 8 rows, selected columns)")
preview_cols = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes", "count",
                "serror_rate", "same_srv_rate", LABEL]
st.dataframe(df[preview_cols].head(8), hide_index=True, width="stretch")

st.markdown("**Step 2 — Run the algorithm** with T = 100 trees, ψ = 256 and the paper's threshold "
            "(score > 0.5). The model uses only the numeric features; the `label` column is hidden "
            "from it and used only afterwards to interpret the results.")

if st.button("▶  Run Isolation Forest", type="primary"):
    st.session_state.result = run_isolation_forest(df)

if "result" in st.session_state:
    result, features, psi = st.session_state.result
    n_anom = int((result["prediction"] == "Anomaly").sum())
    n_norm = len(result) - n_anom

    st.markdown("**Step 3 — Results**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Records analysed", f"{len(result):,}")
    m2.metric("Normal", f"{n_norm:,}")
    m3.metric("Anomalous", f"{n_anom:,}  ({100 * n_anom / len(result):.1f}%)")

    fig = go.Figure()
    for name, color in [("Normal", BLUE), ("Anomaly", ORANGE)]:
        scores = result.loc[result["prediction"] == name, "anomaly_score"]
        fig.add_trace(go.Histogram(x=scores, name=f"{name} ({len(scores):,})", marker_color=color,
                                   xbins=dict(size=0.01), marker_line=dict(color="white", width=1)))
    fig.add_vline(x=0.5, line=dict(color=INK, dash="dash", width=1.5))
    fig.add_annotation(x=0.5, y=1, yref="paper", text="threshold 0.5", showarrow=False,
                       xanchor="left", xshift=6, font=dict(color=MUTED, size=12))
    fig.update_layout(barmode="overlay", height=330, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=40, b=10), font=dict(color=INK),
                      title=dict(text="Anomaly scores of normal and anomalous records", font=dict(size=14)),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    fig.update_xaxes(title="Anomaly score s (higher = more unusual)", gridcolor="#eef0f3")
    fig.update_yaxes(title="Number of records", gridcolor="#eef0f3")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("**Step 4 — The most unusual connections** (top 8 by anomaly score)")
    top = result.sort_values("anomaly_score", ascending=False).head(8)
    table = top[["anomaly_score", "avg_path_length", "protocol_type", "service", "src_bytes",
                 "dst_bytes", "count", "serror_rate", LABEL]].rename(
        columns={"anomaly_score": "score s", "avg_path_length": "avg. path length",
                 LABEL: "true label (not used)"})
    st.dataframe(table.round(3), hide_index=True, width="stretch")

    flagged_attacks = int(((result["prediction"] == "Anomaly") & (result[LABEL] != "normal")).sum())
    flagged_normal = n_anom - flagged_attacks
    st.markdown("**Step 5 — What the results mean**")
    st.markdown(f"""
- The model flagged **{n_anom} of {len(result):,}** connections as anomalies. Their average path length was
  short compared with the typical value c(ψ) = {c(psi):.2f}, so their scores are above 0.5.
- Comparing with the hidden labels afterwards: **{flagged_attacks} of the {n_attack} attacks** were flagged
  ({100 * flagged_attacks / n_attack:.0f}%). The other **{flagged_normal}** flagged connections are labelled
  normal: they are unusual, but not attacks.
- This shows both the strength and the limit of the algorithm: it found most attacks **without ever seeing
  a label**, but "unusual" is not the same as "malicious", so flagged records still need investigation.
""")
    st.caption(f"Model: scikit-learn IsolationForest(n_estimators=100, max_samples=256, contamination='auto', "
               f"random_state=42), trained on {len(features)} numeric features.")

st.divider()
st.caption("**References.** F. T. Liu, K. M. Ting, Z.-H. Zhou, “Isolation Forest,” IEEE ICDM, 2008. · "
           "M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data "
           "Set,” IEEE CISDA, 2009 (NSL-KDD).")
