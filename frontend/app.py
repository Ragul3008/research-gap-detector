"""
frontend/app.py — v3.0
Streamlit UI with:
  - Novelty Analysis
  - Similar Papers
  - Research Gap Report (LLM Enhanced)
  - Plagiarism Detection Dashboard
  - Hallucination Detection
"""

import time
import json
from typing import Dict, Any, Optional

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="AI Research Novelty & Gap Detector",
    page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ─── CSS ─────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size:2.2rem; font-weight:800;
        background:linear-gradient(90deg,#1a237e,#0d47a1,#1565c0);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .sub-header { font-size:1rem; color:#546e7a; margin-bottom:1.2rem; }
    .novelty-badge-high   { background:#2ecc71; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; }
    .novelty-badge-medium { background:#f39c12; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; }
    .novelty-badge-low    { background:#e74c3c; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; }
    .gap-card        { background:#fff3e0; border-left:4px solid #f57c00; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; font-size:0.95rem; }
    .paper-card      { background:#e8f5e9; border-left:4px solid #388e3c; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .title-suggestion{ background:#e3f2fd; border-left:4px solid #1976d2; padding:10px 14px; border-radius:8px; margin:8px 0; font-style:italic; color:#1a1a1a !important; }
    .metric-card     { background:#f8f9fa; border-left:4px solid #1565c0; padding:12px 16px; border-radius:8px; margin:6px 0; color:#1a1a1a !important; }
    .plag-safe       { background:#e8f5e9; border-left:4px solid #2ecc71; padding:12px 16px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .plag-low        { background:#e3f2fd; border-left:4px solid #3498db; padding:12px 16px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .plag-medium     { background:#fff8e1; border-left:4px solid #f39c12; padding:12px 16px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .plag-high       { background:#ffebee; border-left:4px solid #e74c3c; padding:12px 16px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .flagged-sent    { background:#fce4ec; border-left:4px solid #c62828; padding:8px 12px; border-radius:6px; margin:6px 0; color:#1a1a1a !important; font-size:0.88rem; }
    .phrase-match    { background:#f3e5f5; border-left:4px solid #7b1fa2; padding:8px 12px; border-radius:6px; margin:6px 0; color:#1a1a1a !important; font-size:0.88rem; }
    .hallucination-safe   { background:#e8f5e9; border-left:4px solid #2ecc71; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .hallucination-warn   { background:#fff8e1; border-left:4px solid #f39c12; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .hallucination-danger { background:#ffebee; border-left:4px solid #e74c3c; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .flagged-claim   { background:#fce4ec; border-left:4px solid #c62828; padding:8px 12px; border-radius:6px; margin:6px 0; color:#1a1a1a !important; font-size:0.9rem; }
    .llm-badge       { background:#6200ea; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .section-header  { font-size:1.3rem; font-weight:700; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)


# ─── API helpers ─────────────────────────────
def call_api(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Start the FastAPI server on port 8000.")
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Try again.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error {e.response.status_code}: {e.response.text}")
    return None


def get_health():
    try:
        return requests.get(f"{API_BASE}/api/health", timeout=5).json()
    except:
        return {"status": "offline", "index_ready": False, "llm_enabled": False}


# ─── Charts ──────────────────────────────────
def gauge(value, title, suffix, color, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13}},
        number={"suffix": suffix, "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,   max_val*0.4], "color": "#ffebee"},
                {"range": [max_val*0.4, max_val*0.7], "color": "#fff8e1"},
                {"range": [max_val*0.7, max_val],     "color": "#e8f5e9"},
            ],
        }
    ))
    fig.update_layout(height=230, margin=dict(l=15,r=15,t=55,b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plagiarism_gauge(score, color):
    # Inverted — low score = good (green end)
    inv = 100 - score
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "<b>Plagiarism Score</b><br><small>Lower is better</small>",
               "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  15],  "color": "#e8f5e9"},
                {"range": [15, 40],  "color": "#e3f2fd"},
                {"range": [40, 70],  "color": "#fff8e1"},
                {"range": [70, 100], "color": "#ffebee"},
            ],
        }
    ))
    fig.update_layout(height=230, margin=dict(l=15,r=15,t=55,b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def sim_bar(papers):
    titles = [p["title"][:55]+"…" if len(p["title"])>55 else p["title"] for p in papers]
    scores = [p["similarity_pct"] for p in papers]
    colors = ["#e53935" if s>=75 else "#fb8c00" if s>=55 else "#43a047" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=titles, orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores], textposition="outside",
        textfont=dict(color="#1a1a1a", size=12),
    ))
    fig.update_layout(title="Similarity Scores", xaxis_range=[0,110],
                      height=max(250,len(papers)*40),
                      margin=dict(l=10,r=40,t=40,b=20),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#f8f9fa",
                      yaxis=dict(automargin=True))
    return fig


def domain_pie(dd):
    fig = px.pie(names=list(dd.keys()), values=list(dd.values()),
                 title="Domain Distribution",
                 color_discrete_sequence=px.colors.qualitative.Set3, hole=0.35)
    fig.update_layout(height=270, margin=dict(l=5,r=5,t=40,b=5),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def year_bar(yd):
    years  = [str(y) for y in sorted(int(k) for k in yd.keys()) if y > 0]
    counts = [yd.get(y, yd.get(int(y), 0)) for y in years]
    fig = go.Figure(go.Bar(x=years, y=counts, marker_color="#1565c0",
                           text=counts, textposition="outside",
                           textfont=dict(color="#1a1a1a", size=12)))
    fig.update_layout(title="Publication Years", height=240,
                      margin=dict(l=10,r=10,t=40,b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8f9fa")
    return fig


def donut(labels, values, colors, title):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.5,
                           marker_colors=colors))
    fig.update_layout(title=title, height=230,
                      margin=dict(l=5,r=5,t=40,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
    return fig


# ─── Sidebar ─────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 Research Detector v3.0")
        st.divider()
        h = get_health()
        st.markdown("### 📡 System Status")
        if h.get("index_ready"):
            st.success(f"✅ API Online · {h.get('corpus_size',0)} papers")
        else:
            st.error("❌ API Offline")
        st.success("🤖 Gemini LLM" if h.get("llm_enabled") else "⚙️ Template Mode")
        st.success("🛡️ Hallucination Check: ON")
        st.success("🔍 Plagiarism Check: ON")
        st.divider()
        st.markdown("### ⚙️ Settings")
        top_k          = st.slider("Papers to retrieve", 3, 15, 8)
        check_plag     = st.toggle("Enable Plagiarism Check", value=True)
        st.divider()
        st.markdown("### 📖 What This Checks")
        st.markdown("""
- **Novelty** — How original is your topic?
- **Gaps** — What's missing in literature?
- **Plagiarism** — Similarity to existing papers
- **Hallucination** — Are AI claims grounded?
        """)
        st.caption("© 2024 · AI Research Tools · India")
    return top_k, check_plag


# ─── Results renderer ────────────────────────
def render_results(result: Dict[str, Any]):
    novelty       = result.get("novelty", {})
    sim           = result.get("similarity", {})
    papers        = result.get("similar_papers", [])
    gaps          = result.get("gaps", {})
    titles        = result.get("title_suggestions", [])
    explanation   = result.get("explanation", "")
    hallucination = result.get("hallucination", {})
    plagiarism    = result.get("plagiarism", {})

    label   = novelty.get("label", "MEDIUM")
    pct     = novelty.get("percentage", 50)
    n_color = novelty.get("color", "#f39c12")
    h_score = hallucination.get("grounding_score", 100)
    h_color = hallucination.get("color", "#2ecc71")
    p_score = plagiarism.get("plagiarism_score", 0.0)
    p_color = plagiarism.get("color", "#2ecc71")
    o_score = plagiarism.get("originality_score", 100.0)

    # ══ TOP METRICS ═══════════════════════════
    st.markdown("---")
    st.subheader("📊 Analysis Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(gauge(pct, f"<b>Novelty</b><br>{label}", "%", n_color),
                        use_container_width=True, key="ng")
        st.markdown(f'<div style="text-align:center"><span class="novelty-badge-{label.lower()}">{label}</span></div>',
                    unsafe_allow_html=True)
    with c2:
        st.plotly_chart(plagiarism_gauge(p_score, p_color),
                        use_container_width=True, key="pg")
        risk = plagiarism.get("risk_level","SAFE")
        st.markdown(f'<div style="text-align:center;color:{p_color}"><b>{plagiarism.get("icon","")} {risk} RISK</b></div>',
                    unsafe_allow_html=True)
    with c3:
        st.plotly_chart(gauge(o_score, "<b>Originality Score</b>", "%", "#2ecc71"),
                        use_container_width=True, key="og")
    with c4:
        st.plotly_chart(gauge(h_score, "<b>Grounding Score</b><br><small>AI Accuracy</small>", "/100", h_color),
                        use_container_width=True, key="hg")

    # quick metrics row
    stats = sim.get("stats", {})
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Max Similarity",   f"{stats.get('max_sim',0)*100:.1f}%")
    m2.metric("Mean Similarity",  f"{stats.get('mean_sim',0)*100:.1f}%")
    m3.metric("Papers Retrieved", len(papers))
    m4.metric("Sentences Flagged",plagiarism.get("stats",{}).get("sentences_flagged",0))

    st.markdown(f'<div class="metric-card">{novelty.get("description","")}</div>',
                unsafe_allow_html=True)

    if sim.get("duplicate_risk"):
        st.error(f"🚨 **Duplicate Risk!** {len(sim.get('duplicates',[]))} paper(s) ≥90% similar.")

    # sub-scores
    sub = novelty.get("sub_scores", {})
    if sub:
        with st.expander("🔢 Novelty Sub-Score Breakdown"):
            s1,s2,s3 = st.columns(3)
            s1.metric("Similarity", f"{sub.get('similarity_sub',0)*100:.0f}%")
            s2.metric("Coverage",   f"{sub.get('coverage_sub',0)*100:.0f}%")
            s3.metric("Recency",    f"{sub.get('recency_sub',0)*100:.0f}%")

    # ══ PLAGIARISM SECTION ════════════════════
    st.markdown("---")
    st.subheader("🔍 Plagiarism Detection Report")

    risk    = plagiarism.get("risk_level", "SAFE")
    css_map = {"SAFE":"plag-safe","LOW":"plag-low","MEDIUM":"plag-medium","HIGH":"plag-high"}
    css     = css_map.get(risk, "plag-safe")

    st.markdown(
        f'<div class="{css}"><b>{plagiarism.get("icon","")} {risk} PLAGIARISM RISK — '
        f'{p_score:.1f}% Similar | {o_score:.1f}% Original</b><br>'
        f'{plagiarism.get("message","")}</div>',
        unsafe_allow_html=True
    )

    # Plagiarism stats row
    ps = plagiarism.get("stats", {})
    pc1,pc2,pc3,pc4 = st.columns(4)
    pc1.metric("Plagiarism Score",    f"{p_score:.1f}%")
    pc2.metric("Originality Score",   f"{o_score:.1f}%")
    pc3.metric("Sentences Checked",   ps.get("total_sentences_checked", 0))
    pc4.metric("Phrases Flagged",     ps.get("phrases_flagged", 0))

    # Matched papers
    matched = plagiarism.get("matched_papers", [])
    if matched:
        with st.expander("📄 Matched Papers (Similarity Breakdown)"):
            for mp in matched:
                risk_color = "#e74c3c" if mp["risk"]=="HIGH" else "#f39c12" if mp["risk"]=="MEDIUM" else "#3498db"
                st.markdown(
                    f'<div class="paper-card">'
                    f'<b style="color:{risk_color}">[{mp["risk"]}]</b> '
                    f'{mp["title"]}<br>'
                    f'<small>Year: {mp["year"]} · Domain: {mp["domain"]} · '
                    f'Similarity: <b>{mp["similarity_pct"]:.1f}%</b></small>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Sentence-level matches
    sent_matches = plagiarism.get("sentence_matches", [])
    if sent_matches:
        with st.expander(f"⚠️ Flagged Sentences ({len(sent_matches)} found)"):
            for sm in sent_matches:
                st.markdown(
                    f'<div class="flagged-sent">'
                    f'<b>Your text:</b> "{sm.get("user_sentence","")[:150]}"<br>'
                    f'<b>Matches:</b> "{sm.get("paper_sentence","")[:150]}"<br>'
                    f'<small>📄 {sm.get("paper_title","")[:60]} '
                    f'({sm.get("paper_year","")}) — '
                    f'<b>{sm.get("similarity_pct",0):.1f}% similar</b></small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.success("✅ No sentence-level matches found.")

    # Exact phrase matches
    phrase_matches = plagiarism.get("phrase_matches", [])
    if phrase_matches:
        with st.expander(f"🔤 Exact Phrase Matches ({len(phrase_matches)} found)"):
            for pm in phrase_matches:
                st.markdown(
                    f'<div class="phrase-match">'
                    f'🔤 <b>"{pm.get("phrase","")}"</b> '
                    f'({pm.get("word_count",0)} words) — '
                    f'Found in: {pm.get("paper_title","")[:60]} ({pm.get("paper_year","")})'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.success("✅ No exact phrase matches found.")

    # Plagiarism recommendations
    plag_recs = plagiarism.get("recommendations", [])
    if plag_recs:
        with st.expander("💡 How to Fix Plagiarism Issues"):
            for r in plag_recs:
                st.markdown(f"- {r}")

    # ══ SIMILAR PAPERS ════════════════════════
    st.markdown("---")
    st.subheader("📚 Most Similar Research Papers")
    if papers:
        st.plotly_chart(sim_bar(papers), use_container_width=True, key="sb")
        for p in papers[:5]:
            with st.expander(f"📄 [{p.get('rank','?')}] {p.get('title','')} — {p.get('similarity_pct',0):.1f}% similar"):
                c1,c2,c3 = st.columns(3)
                c1.markdown(f"**Year:** {p.get('year','N/A')}")
                c2.markdown(f"**Domain:** {p.get('domain','N/A')}")
                c3.markdown(f"**Region:** {p.get('region','N/A')}")
                st.markdown(f"**Abstract:** {p.get('abstract','')[:400]}…")
                st.markdown(f"**Keywords:** _{p.get('keywords','')}_")
    else:
        st.info("No similar papers found — your topic may be highly novel!")

    # ══ DISTRIBUTION CHARTS ════════════════════
    if papers:
        st.markdown("---")
        cd,cy = st.columns(2)
        with cd:
            st.plotly_chart(domain_pie(sim.get("domain_dist",{})),
                            use_container_width=True, key="dp")
        with cy:
            st.plotly_chart(year_bar(sim.get("year_dist",{})),
                            use_container_width=True, key="yb")

    # ══ GAP REPORT ════════════════════════════
    st.markdown("---")
    st.subheader("🔍 Research Gap Report")
    from config.settings import USE_LLM_EXPLANATION
    if USE_LLM_EXPLANATION:
        st.markdown('<span class="llm-badge">✨ Enhanced by Gemini LLM</span>',
                    unsafe_allow_html=True)
        st.markdown("")

    for stmt in gaps.get("gap_statements", []):
        st.markdown(f'<div class="gap-card">{stmt}</div>', unsafe_allow_html=True)

    gap_dims = gaps.get("gap_dimensions", {})
    if gap_dims:
        with st.expander("📋 Detailed Gap Dimensions"):
            d1,d2 = st.columns(2)
            with d1:
                for key, label in [("regional_gaps","📍 Regions"),("population_gaps","👥 Populations"),("temporal_gaps","📅 Time Periods")]:
                    items = gap_dims.get(key,[])
                    if items:
                        st.markdown(f"**{label}:**")
                        for g in items[:5]: st.markdown(f"  - {g.title()}")
            with d2:
                for key, label in [("methodological_gaps","🔬 Methods"),("thematic_gaps","💡 Themes"),("theoretical_gaps","📚 Theories")]:
                    items = gap_dims.get(key,[])
                    if items:
                        st.markdown(f"**{label}:**")
                        for g in items[:5]: st.markdown(f"  - {g.title()}")

    # ══ TITLE SUGGESTIONS ═════════════════════
    st.markdown("---")
    st.subheader("✍️ Suggested Improved Research Titles")
    for i,t in enumerate(titles,1):
        st.markdown(f'<div class="title-suggestion">📝 <b>Option {i}:</b> {t}</div>',
                    unsafe_allow_html=True)

    # ══ RECOMMENDATIONS ═══════════════════════
    sugg = novelty.get("suggestions",[])
    if sugg:
        st.markdown("---")
        st.subheader("💡 Actionable Recommendations")
        for s in sugg: st.markdown(f"- {s}")

    # ══ AI EXPLANATION ════════════════════════
    st.markdown("---")
    st.subheader("📋 AI-Generated Research Report")
    if USE_LLM_EXPLANATION:
        st.markdown('<span class="llm-badge">🤖 Generated by Gemini · Grounded in Evidence</span>',
                    unsafe_allow_html=True)
        st.markdown("")
    with st.expander("📖 View Full Report", expanded=True):
        st.markdown(explanation.replace("\n","  \n"))

    # ══ HALLUCINATION SECTION ═════════════════
    st.markdown("---")
    st.subheader("🛡️ Hallucination Detection Report")

    h_count = hallucination.get("hallucination_count", 0)
    h_total = hallucination.get("total_claims", 0)

    hc1,hc2,hc3,hc4 = st.columns(4)
    hc1.metric("Grounding Score", f"{h_score}/100")
    hc2.metric("Total Claims",    h_total)
    hc3.metric("Grounded",        h_total - h_count)
    hc4.metric("Flagged",         h_count)

    h_css = "hallucination-safe" if h_score>=80 else "hallucination-warn" if h_score>=60 else "hallucination-danger"
    st.markdown(
        f'<div class="{h_css}"><b>{hallucination.get("label","")}</b> — '
        f'Grounding Score: {h_score}/100</div>',
        unsafe_allow_html=True
    )

    if h_total > 0:
        col_d, col_w = st.columns([1,2])
        with col_d:
            st.plotly_chart(
                donut(["Grounded","Flagged"],[h_total-h_count,h_count],
                      ["#2ecc71","#e74c3c"],"Claim Verification"),
                use_container_width=True, key="hd"
            )
        with col_w:
            warnings = hallucination.get("warnings",[])
            if warnings:
                for w in warnings: st.warning(w)
            else:
                st.success("✅ No warnings — explanation is well-grounded.")

    flagged = hallucination.get("flagged_claims",[])
    if flagged:
        with st.expander(f"🔴 View {len(flagged)} Flagged Claim(s)"):
            for fc in flagged:
                sev  = fc.get("severity","LOW")
                icon = "🔴" if sev=="HIGH" else "🟡" if sev=="MEDIUM" else "🟢"
                st.markdown(
                    f'<div class="flagged-claim">{icon} <b>[{sev}]</b> {fc.get("issue","")}<br>'
                    f'<small><i>"{fc.get("claim","")[:120]}…"</i></small></div>',
                    unsafe_allow_html=True
                )
    else:
        st.success("✅ No hallucinations detected.")

    # ══ DOWNLOAD ══════════════════════════════
    st.markdown("---")
    st.download_button(
        label="📥 Download Full Analysis (JSON)",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="research_analysis.json",
        mime="application/json",
        use_container_width=True,
    )


# ─── Main ────────────────────────────────────
def main():
    top_k, check_plag = render_sidebar()

    st.markdown('<p class="main-header">🔬 AI Research Novelty & Gap Detector</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">For Arts, Science & Engineering PhD Scholars in India · '
        'Gemini LLM + FAISS + Plagiarism Detection + Hallucination Check</p>',
        unsafe_allow_html=True
    )

    # Initialize session state for extracted metadata
    if "extracted_title" not in st.session_state:
        st.session_state.extracted_title = ""
    if "extracted_abstract" not in st.session_state:
        st.session_state.extracted_abstract = ""
    if "extracted_keywords" not in st.session_state:
        st.session_state.extracted_keywords = ""
    if "extracted_domain" not in st.session_state:
        st.session_state.extracted_domain = ""
    if "extracted_full_text" not in st.session_state:
        st.session_state.extracted_full_text = ""

    # Option 1: File Upload
    st.subheader("📁 Option 1: Upload a Journal Paper (PDF / TXT)")
    uploaded_file = st.file_uploader("Upload your journal paper to automatically extract Title, Abstract, Keywords, and Domain", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_file") != uploaded_file.name:
            with st.spinner("⏳ Extracting metadata from paper..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    r = requests.post(f"{API_BASE}/api/extract-metadata", files=files, timeout=60)
                    r.raise_for_status()
                    res = r.json()
                    if res.get("status") == "success":
                        meta = res.get("metadata", {})
                        st.session_state.extracted_title = meta.get("title", "")
                        st.session_state.extracted_abstract = meta.get("abstract", "")
                        st.session_state.extracted_keywords = meta.get("keywords", "")
                        st.session_state.extracted_domain = meta.get("domain", "")
                        st.session_state.extracted_full_text = res.get("full_text", "")
                        st.session_state.last_uploaded_file = uploaded_file.name
                        st.success("✅ Metadata extracted! Review the populated fields below.")
                except Exception as e:
                    st.error(f"❌ Failed to extract metadata: {e}")
    else:
        if "last_uploaded_file" in st.session_state:
            st.session_state.extracted_title = ""
            st.session_state.extracted_abstract = ""
            st.session_state.extracted_keywords = ""
            st.session_state.extracted_domain = ""
            st.session_state.extracted_full_text = ""
            del st.session_state.last_uploaded_file

    st.markdown("---")

    # Option 2: Enter/Review details form
    with st.form("form"):
        st.subheader("📝 Option 2: Review & Edit Paper Details")
        col1, col2 = st.columns([3, 1])
        with col1:
            title = st.text_input("Research Title *",
                value=st.session_state.extracted_title,
                placeholder="e.g., Deep Learning for Medical Image Segmentation in Indian Hospitals")
        with col2:
            domain_options = [
                "",
                # Arts & Science
                "Sociology", "History", "Political Science", "Economics",
                "Psychology", "Environmental Science", "Literature", "Philosophy",
                "Anthropology", "Education", "Geography", "Women Studies",
                "Commerce", "Linguistics", "Botany", "Zoology",
                "Chemistry", "Physics", "Mathematics", "Statistics",
                # English
                "English Literature", "English Language Teaching",
                "Postcolonial Studies", "Comparative Literature", "Translation Studies",
                # Engineering
                "Computer Science and Engineering", "Electronics and Communication Engineering",
                "Electrical Engineering", "Mechanical Engineering", "Civil Engineering",
                "Chemical Engineering", "Aerospace Engineering", "Biomedical Engineering",
                "Artificial Intelligence", "Machine Learning", "Deep Learning",
                "Internet of Things", "Cybersecurity", "Data Science",
                "Robotics and Automation", "VLSI Design", "Power Systems Engineering",
                "Renewable Energy Engineering", "Structural Engineering",
                "Environmental Engineering", "Nanotechnology", "Materials Science",
            ]
            default_idx = 0
            if st.session_state.extracted_domain in domain_options:
                default_idx = domain_options.index(st.session_state.extracted_domain)
            domain = st.selectbox("Domain", domain_options, index=default_idx)

        abstract = st.text_area("Abstract (recommended)",
            value=st.session_state.extracted_abstract,
            placeholder="Describe your research objectives, methodology, and expected contribution…",
            height=120)
        keywords = st.text_input("Keywords (comma-separated)",
            value=st.session_state.extracted_keywords,
            placeholder="e.g., deep learning, medical imaging, CNN, India")

        submitted = st.form_submit_button("🚀 Analyse Research", use_container_width=True)

    if submitted:
        if not title.strip():
            st.warning("⚠️ Please enter a research title.")
            return

        with st.spinner("🔍 Running full analysis pipeline…"):
            t0 = time.time()
            result = call_api("/api/analyze", {
                "title":            title.strip(),
                "abstract":         abstract.strip(),
                "keywords":         keywords.strip(),
                "domain":           domain or None,
                "top_k":            top_k,
                "check_plagiarism": check_plag,
                "full_text":        st.session_state.extracted_full_text or None,
            })

        if result is None:
            return

        elapsed = time.time() - t0
        st.success(f"✅ Full analysis complete in {elapsed:.1f}s")
        render_results(result)


if __name__ == "__main__":
    main()