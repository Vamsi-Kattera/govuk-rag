"""Day 6: Streamlit front end. Talks to the FastAPI service over HTTP.

Run locally (two terminals, from the repo root):
    python -m uvicorn api:app --port 8000
    python -m streamlit run ui.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
EXAMPLES = [
    "How long is an eligible employee entitled to take for Statutory Maternity Leave?",
    "Can I travel abroad on a Graduate visa?",
    "What is the National Minimum Wage for a 21 year old?",
    "Who won the cricket world cup?",  # out of scope: should be refused
]

st.set_page_config(page_title="GOV.UK RAG assistant", page_icon="🇬🇧", layout="centered")
st.title("GOV.UK RAG assistant")
st.caption(
    "Answers questions about UK visas, right to work, employment rights and benefits, "
    "using only ~130 indexed GOV.UK pages. Every answer is cited, and it says "
    "\"I don't know\" when the pages don't cover the question."
)

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Retrieval mode",
        ["rerank", "hybrid", "bm25", "vector"],
        help="rerank = hybrid search + cross-encoder reranking (best in my evaluation).",
    )
    st.markdown("**Try an example**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex
    st.markdown("---")
    st.caption(
        "Contains public sector information licensed under the "
        "[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). "
        "Not legal advice."
    )


def ask(question: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/ask", json={"question": question, "mode": mode}, timeout=60
        )
    except requests.RequestException:
        return {"error": "Could not reach the API. Is the backend running?"}
    if r.status_code == 200:
        return r.json()
    if r.status_code == 429:
        return {"error": "Too many requests - please wait a minute and try again."}
    if r.status_code == 422:
        return {"error": "Please enter a question between 3 and 500 characters."}
    return {"error": "The service is temporarily unavailable. Please try again shortly."}


def show(msg: dict) -> None:
    st.markdown(msg["content"])
    d = msg.get("data")
    if not d:
        return
    if d.get("sources"):
        with st.expander(f"Sources ({len(d['sources'])})"):
            for s in d["sources"]:
                st.markdown(f"**[{s['n']}]** [{s['title']}]({s['url']})  \nLast updated: {s['updated'][:10]}")
    st.caption(
        f"{'answered' if d['answered'] else 'refused'} · similarity {d['best_similarity']} · "
        f"retrieval {d['retrieve_ms']} ms · total {d['total_ms']} ms"
    )


if "history" not in st.session_state:
    st.session_state["history"] = []

for m in st.session_state["history"]:
    with st.chat_message(m["role"]):
        show(m)

question = st.chat_input("Ask about UK visas, right to work, benefits...") or st.session_state.pop("pending", None)
if question:
    st.session_state["history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching GOV.UK pages..."):
            res = ask(question)
        if "error" in res:
            msg = {"role": "assistant", "content": res["error"]}
        else:
            msg = {"role": "assistant", "content": res["answer"], "data": res}
        show(msg)
    st.session_state["history"].append(msg)
