import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load configuration before importing modules that read API keys at import time.
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Echobrief | Video Intelligence",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM UI STYLING (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Gradient text for main title */
    .hero-title {
        font-size: 3.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #8E2DE2, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.5rem;
        margin-bottom: 0;
        text-align: center;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.8;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 400;
    }
    /* Adjust top padding for a cleaner look */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SESSION STATE & HELPERS
# ---------------------------------------------------------
DEFAULTS = {
    "result": None,
    "chat_history": [],
    "error": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

def reset_session():
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.error = None

def run_pipeline_with_progress(source: str, language: str) -> dict:
    with st.status("Processing your video...", expanded=True) as status:
        def step(label: str):
            status.write(f":material/check_circle: {label}")

        step("Ingesting media source")
        chunks = process_input(source)

        step("Transcribing audio tracks")
        transcript = transcribe_all(chunks, language)

        step("Generating executive summary & title")
        title = generate_title(transcript)
        summary = summarize(transcript)

        # Note: Extraction runs in the background to feed the RAG/context if needed, 
        # even if not displayed on the front-end tabs.
        step("Extracting insights")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        step("Initializing intelligent chat (RAG)")
        rag_chain = build_rag_chain(transcript)
        
        status.update(label="Echobrief successfully generated", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("Echobrief")
    st.badge("VIDEO INTELLIGENCE", color="violet")
    st.space("small")
    st.caption("Turn lengthy recordings into clear, actionable intelligence.")
    
    st.divider()
    
    st.caption("WORKFLOW")
    st.write(":material/link: **1.** Connect media")
    st.write(":material/auto_awesome: **2.** Generate brief")
    st.write(":material/forum: **3.** Chat with context")
    
    st.space("large")
    clear_clicked = st.button("New Echobrief", icon=":material/add:", width="stretch", type="secondary")
    if clear_clicked:
        reset_session()
        st.rerun()
        
    st.divider()
    st.caption(":material/lock: All processing is secure and API keys remain local.")


# ---------------------------------------------------------
# MAIN APP LOGIC
# ---------------------------------------------------------
process_clicked = False
input_mode = "YouTube URL"
source_value = None
uploaded_file = None
language = "english"

# === STATE 1: INPUT SCREEN ===
if st.session_state.result is None:
    st.markdown('<div class="hero-title">Turn any recording into<br>a brief worth reading.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Echobrief finds the signal in meetings, interviews, and explainers—extracting the context that matters.</div>', unsafe_allow_html=True)
    st.space("medium")

    left_space, form_column, right_space = st.columns([1.5, 3, 1.5])
    with form_column:
        with st.container(border=True):
            st.subheader("Create your brief", anchor=False)
            st.caption("Paste a YouTube link or upload a local file to begin.")
            st.space("small")
            
            with st.form("source_form", border=False):
                input_mode = st.segmented_control(
                    "Source type",
                    ["YouTube URL", "Upload a file"],
                    default="YouTube URL",
                    label_visibility="collapsed"
                )
                
                st.space("small")
                
                if input_mode == "YouTube URL":
                    source_value = st.text_input(
                        "YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        label_visibility="collapsed"
                    )
                else:
                    uploaded_file = st.file_uploader(
                        "Audio or video file",
                        type=["mp4", "mov", "mkv", "mp3", "wav", "m4a", "webm"],
                        label_visibility="collapsed"
                    )
                    
                language = st.selectbox(
                    "Spoken language",
                    ["english", "hinglish"],
                    format_func=lambda option: option.title(),
                )
                
                st.space("small")
                process_clicked = st.form_submit_button(
                    "Generate Intelligence",
                    icon=":material/auto_awesome:",
                    type="primary",
                    width="stretch",
                )

    st.space("large")
    details = st.columns(3, gap="large")
    details[0].markdown("**1. Accurate**<br><span style='font-size:0.9em; opacity:0.7;'>High-fidelity transcription</span>", unsafe_allow_html=True)
    details[1].markdown("**2. Actionable**<br><span style='font-size:0.9em; opacity:0.7;'>Comprehensive summaries</span>", unsafe_allow_html=True)
    details[2].markdown("**3. Interactive**<br><span style='font-size:0.9em; opacity:0.7;'>Grounded Q&A chat</span>", unsafe_allow_html=True)

# === TRIGGER PIPELINE ===
if process_clicked:
    reset_session()
    resolved_source = None
    if input_mode == "YouTube URL":
        if not source_value or not source_value.strip():
            st.session_state.error = "Please provide a valid YouTube URL."
        else:
            resolved_source = source_value.strip()
    elif uploaded_file is None:
        st.session_state.error = "Please upload an audio or video file."
    else:
        tmp_dir = tempfile.mkdtemp(prefix="echobrief_")
        tmp_path = Path(tmp_dir) / uploaded_file.name
        with open(tmp_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        resolved_source = str(tmp_path)

    if resolved_source:
        try:
            st.session_state.result = run_pipeline_with_progress(resolved_source, language)
            st.toast("Echobrief successfully generated!", icon=":material/check_circle:")
            st.rerun() 
        except Exception as error:
            st.session_state.error = f"Generation failed: {error}"


# === ERROR HANDLING ===
if st.session_state.error:
    st.error(st.session_state.error, icon=":material/error:")

# === STATE 2: RESULT SCREEN ===
result = st.session_state.result
if result is not None:
    st.badge("ECHOBRIEF COMPLETE", icon=":material/check:", color="green")
    st.header(result["title"], anchor=False)
    st.caption("A concise intelligence layer extracted from your recording.")
    st.space("small")

    # Metrics Layout
    transcript_words = len(result["transcript"].split())
    with st.container(border=True):
        st.metric("Transcript Length", f"{transcript_words:,} words")

    st.space("medium")

    # Tabs Layout (Updated to only show 3 requested tabs)
    summary_tab, transcript_tab, chat_tab = st.tabs(
        ["📋 Overview", "📝 Transcript", "💬 Ask Echobrief"]
    )

    with summary_tab:
        with st.container(border=True):
            st.subheader("Executive Summary", anchor=False)
            st.markdown(result["summary"])
        st.download_button(
            "Download Summary",
            data=result["summary"],
            file_name="echobrief-summary.txt",
            mime="text/plain",
            icon=":material/download:",
        )

    with transcript_tab:
        with st.container(border=True):
            st.subheader("Full Transcript", anchor=False)
            st.text_area(
                "Full transcript",
                result["transcript"],
                height=400,
                label_visibility="collapsed",
            )
        st.download_button(
            "Download Transcript",
            data=result["transcript"],
            file_name="echobrief-transcript.txt",
            mime="text/plain",
            icon=":material/download:",
        )

    with chat_tab:
        st.caption("Ask a question in natural language. Each answer is accurately grounded in the transcript.")
        with st.container(border=True):
            # Chat history rendering
            if not st.session_state.chat_history:
                st.info("💡 Try asking: “Summarize the main argument” or “What was mentioned about the budget?”", icon=":material/lightbulb:")
            
            chat_container = st.container(height=400, border=False)
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

        # Chat input handling
        question = st.chat_input("Ask Echobrief about this video...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.rerun() 

        # Generate assistant response
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with st.spinner("Analyzing transcript..."):
                try:
                    last_question = st.session_state.chat_history[-1]["content"]
                    answer = ask_question(result["rag_chain"], last_question)
                except Exception as error:
                    answer = f"Unable to answer this question: {error}"
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()