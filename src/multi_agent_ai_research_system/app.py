import time

import streamlit as st

from multi_agent_ai_research_system.agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    writer_chain,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 9rem; max-width: 820px; }

/* ── Hero (empty state) ── */
.hero { text-align: center; padding: 3rem 0 2rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #ff8c32;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.6rem, 6vw, 4.2rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f0ebe0;
    margin: 0 0 1rem;
}
.hero h1 span { color: #ff8c32; }
.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    color: #a09890;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    line-height: 1.75;
    color: #d8d3ca;
}

/* ── Status ("thinking") panel ── */
[data-testid="stStatus"], details {
    background: rgba(255,140,50,0.04) !important;
    border: 1px solid rgba(255,140,50,0.2) !important;
    border-radius: 12px !important;
}
details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #a09890 !important;
    letter-spacing: 0.05em !important;
}

/* ── Chat input pinned to bottom ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: transparent !important;
}
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #ff8c32 !important;
    box-shadow: 0 0 0 3px rgba(255,140,50,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] button {
    background: #ff8c32 !important;
    color: #0a0a0f !important;
    border-radius: 10px !important;
}

/* ── Example chips + download button ── */
.stButton > button, .stDownloadButton > button {
    background: rgba(255,255,255,0.04) !important;
    color: #cdc8bf !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s, transform 0.15s !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #ff8c32 !important;
    color: #ff8c32 !important;
    transform: translateY(-1px);
}

/* ── Critic box ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(80,200,120,0.25) !important;
    border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def stream_text(chunks):
    """Turn LangChain stream chunks (str or AIMessageChunk) into plain text pieces."""
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
        if isinstance(text, list):  # some models return content blocks
            text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in text
            )
        if text:
            yield text


def render_user(content: str):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(content)


def render_assistant(msg: dict, idx: int):
    """Render a finished assistant message from history."""
    with st.chat_message("assistant", avatar="🔬"):
        with st.expander("⚙️ Research steps", expanded=False):
            tab_search, tab_reader = st.tabs(["🔍 Search", "📄 Scraped"])
            with tab_search:
                st.markdown(msg["search"])
            with tab_reader:
                st.markdown(msg["reader"])

        st.markdown(msg["report"])
        st.download_button(
            "⬇  Download Report (.md)",
            data=msg["report"],
            file_name=f"research_report_{msg['ts']}.md",
            mime="text/markdown",
            key=f"dl_{idx}",
        )

        with st.container(border=True):
            st.markdown("**🧐 Critic Feedback**")
            st.markdown(msg["critic"])


def set_pending(prompt: str):
    st.session_state.pending_prompt = prompt


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# chat_input is pinned to the bottom no matter where it's called
typed = st.chat_input("Research any topic…")
prompt = typed or st.session_state.pop("pending_prompt", None)


# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages and not prompt:
    st.markdown(
        """
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">
        Four specialized AI agents collaborate — searching, scraping, writing,
        and critiquing — to deliver a polished research report on any topic.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, ex in zip(
        cols, ["LLM agents 2026", "CRISPR gene editing", "Fusion energy progress"]
    ):
        col.button(ex, on_click=set_pending, args=(ex,), use_container_width=True)


# ── Chat history ──────────────────────────────────────────────────────────────
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        render_user(m["content"])
    else:
        render_assistant(m, i)


# ── New prompt → run pipeline live ────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_user(prompt)

    with st.chat_message("assistant", avatar="🔬"):
        try:
            # Steps 1 & 2 inside a collapsible "thinking" panel
            with st.status("Starting research…", expanded=True) as status:
                # ── Step 1: Search ──
                status.update(label="🔍 Search Agent is gathering information…")
                st.write("🔍 **Search Agent** — gathering recent web information…")
                search_agent = build_search_agent()
                sr = search_agent.invoke(
                    {
                        "messages": [
                            (
                                "user",
                                f"Find recent, reliable and detailed information about: {prompt}",
                            )
                        ]
                    }
                )
                search_out = sr["messages"][-1].content
                st.write("✅ Search complete")

                # ── Step 2: Reader ──
                status.update(label="📄 Reader Agent is scraping top resources…")
                st.write("📄 **Reader Agent** — scraping & extracting deep content…")
                reader_agent = build_reader_agent()
                rr = reader_agent.invoke(
                    {
                        "messages": [
                            (
                                "user",
                                f"Based on the following search results about '{prompt}', "
                                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                                f"Search Results:\n{search_out[:800]}",
                            )
                        ]
                    }
                )
                reader_out = rr["messages"][-1].content
                st.write("✅ Scraping complete")

                status.update(
                    label="✅ Research complete", state="complete", expanded=False
                )

            # ── Step 3: Writer (streams token by token) ──
            research_combined = (
                f"SEARCH RESULTS:\n{search_out}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{reader_out}"
            )
            report = st.write_stream(
                stream_text(
                    writer_chain.stream(
                        {"topic": prompt, "research": research_combined}
                    )
                )
            )

            ts = int(time.time())
            st.download_button(
                "⬇  Download Report (.md)",
                data=report,
                file_name=f"research_report_{ts}.md",
                mime="text/markdown",
                key=f"dl_{len(st.session_state.messages)}",
            )

            # ── Step 4: Critic (streams too) ──
            with st.container(border=True):
                st.markdown("**🧐 Critic Feedback**")
                critic = st.write_stream(
                    stream_text(critic_chain.stream({"report": report}))
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "search": search_out,
                    "reader": reader_out,
                    "report": report,
                    "critic": critic,
                    "ts": ts,
                }
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
