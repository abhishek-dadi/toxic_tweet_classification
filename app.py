"""
app.py: Streamlit demo for the toxicity project (predictions only)
==================================================================
Type a comment or tweet and both fine-tuned models (BERT and Twitter-RoBERTa)
classify it side by side. A batch mode classifies several comments at once.

Needs the output folder created by toxicity_stats.py (models/ and thresholds.json).

Run in Colab:   see README ("Streamlit demo")
Run locally :   streamlit run app.py -- --out_dir path/to/aps_toxicity
"""
import argparse
import json
import os
import re
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default=os.environ.get("APS_OUT_DIR", "/content/drive/MyDrive/aps_toxicity"))
ARGS, _ = parser.parse_known_args()

MODELS = {"baseline": ("BERT-base", "bert-base-uncased"),
          "improved": ("Twitter-RoBERTa", "twitter-roberta-base-2022-154m")}

EXAMPLES = {
    "Friendly": "Congrats on the launch, the new feature looks great!",
    "Disagreement": "I completely disagree with this article, the author ignores the data.",
    "Rude": "This take is dumb, honestly.",
    "Insult": "You are a worthless idiot and nobody cares what you think.",
    "Threat": "I know where you live and I will make you regret this.",
    "Slang": "lmao this ref is trash, worst call of the season",
}


def clean(text):
    t = str(text)
    t = re.sub(r"https?://\S+|www\.\S+", "http", t)
    t = re.sub(r"@\w+", "@user", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_resource(show_spinner="Loading both models ...")
def load_models(out_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {}
    for role, (_, folder) in MODELS.items():
        path = f"{out_dir}/models/{folder}"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        loaded[role] = (tok, model)
    return loaded, device


def predict(loaded, device, texts):
    import torch
    out = {}
    for role, (tok, model) in loaded.items():
        t0 = time.perf_counter()
        with torch.inference_mode():
            enc = tok([clean(t) for t in texts], truncation=True, max_length=128,
                      padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
        out[role] = (probs, (time.perf_counter() - t0) * 1000)
    return out


# ----------------------------------------------------------------------------- #
# Sidebar
# ----------------------------------------------------------------------------- #
st.sidebar.title("🛡️ Toxic Comment Classifier")
out_dir = st.sidebar.text_input("Output folder", ARGS.out_dir)
thr_path = f"{out_dir}/thresholds.json"
if os.path.exists(thr_path):
    with open(thr_path) as f:
        thresholds = json.load(f)
else:
    thresholds = {"baseline": 0.5, "improved": 0.5}
st.sidebar.markdown("**Decision thresholds**")
for role, (name, _) in MODELS.items():
    st.sidebar.write(f"{name}: {thresholds[role]:.2f}")
st.sidebar.caption("A comment is flagged as toxic when the model's probability is at or above its threshold.")

# ----------------------------------------------------------------------------- #
# Main page
# ----------------------------------------------------------------------------- #
st.title("Toxic Comment Classifier")
st.write("Type a comment or tweet, and both models will classify it.")

models_ok = all(os.path.exists(f"{out_dir}/models/{folder}/config.json") for _, folder in MODELS.values())
if not models_ok:
    st.error(f"Trained models not found in {out_dir}/models/. Run toxicity_stats.py first, "
             "or fix the output folder in the sidebar.")
    st.stop()

loaded, device = load_models(out_dir)
st.caption(f"Both models loaded on **{device.upper()}**.")

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES["Insult"]


def set_text(t):
    st.session_state.text = t


st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (name, ex) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=set_text, args=(ex,), use_container_width=True)

text = st.text_area("Comment or tweet", key="text", height=100)
if st.button("Classify", type="primary") and text.strip():
    res = predict(loaded, device, [text])
    c1, c2 = st.columns(2)
    for col, role in zip((c1, c2), MODELS):
        p, ms = res[role][0][0], res[role][1]
        toxic = p >= thresholds[role]
        with col:
            st.subheader(MODELS[role][0])
            if toxic:
                st.error(f"🚫 TOXIC  ({p:.1%})")
            else:
                st.success(f"✅ Not toxic  ({p:.1%})")
            st.progress(float(p))
            st.caption(f"Threshold {thresholds[role]:.2f} · {ms:.0f} ms")

st.markdown("---")
st.subheader("Batch mode")
batch = st.text_area("One comment per line", "\n".join(list(EXAMPLES.values())[:4]), height=130)
if st.button("Classify all"):
    lines = [l for l in batch.splitlines() if l.strip()]
    if lines:
        res = predict(loaded, device, lines)
        df = pd.DataFrame({"Comment": lines})
        for role, (name, _) in MODELS.items():
            df[f"{name}: P(toxic)"] = [f"{p:.1%}" for p in res[role][0]]
            df[f"{name}: verdict"] = ["🚫 Toxic" if p >= thresholds[role] else "✅ Not toxic" for p in res[role][0]]
        st.dataframe(df, use_container_width=True, hide_index=True)
