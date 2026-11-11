import re
import time
import unicodedata
from difflib import SequenceMatcher

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from audio_processor import ZikrAudioProcessor

st.set_page_config(
    page_title="Smart Zikr Counter",
    page_icon="🕌",
    layout="centered",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(160deg, #071f1a 0%, #0d3329 55%, #10271f 100%); color: #f7f3e8; }
h1, h2, h3, p, label { color: #f7f3e8 !important; }
.counter {
    text-align:center; font-size: min(24vw, 112px); font-weight:800;
    color:#e8c878; line-height:1.1; padding:18px 0 4px 0;
}
.subtle { text-align:center; color:#c6d5ca; font-size:14px; }
div[data-testid="stMetric"] { background:#164438; padding:14px; border-radius:14px; }
.stButton button { border-radius:12px; min-height:44px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🕌 Smart Zikr Counter")
st.caption("Live voice-based repetition counter • Arabic & Bangla")

# Session state
defaults = {
    "count": 0,
    "reference": None,
    "last_text": "",
    "last_status": "মাইক্রোফোন চালু করে পড়া শুরু করুন।",
    "history": [],
    "candidate": None,
    "processed": 0,
    "target": 100,
    "threshold": 0.72,
    "model": None,
    "model_error": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    # Arabic diacritics, tatweel and punctuation
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ى", "ي"))
    # Keep letters/numbers from all scripts; discard punctuation
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def similarity(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    # Character similarity works as a simple first-pass matcher.
    return SequenceMatcher(None, a, b).ratio()

@st.cache_resource(show_spinner="Speech recognition model is loading (first run may take a few minutes)...")
def load_model():
    from faster_whisper import WhisperModel
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe(audio):
    model = load_model()
    segments, info = model.transcribe(
        audio,
        language=None,          # automatic language detection
        task="transcribe",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, getattr(info, "language", "unknown")

def accept_phrase(text, language):
    cleaned = normalize_text(text)
    if not cleaned:
        st.session_state.last_status = "স্পষ্ট উচ্চারণ শনাক্ত হয়নি। আবার চেষ্টা করুন।"
        return

    st.session_state.last_text = text

    if st.session_state.reference is None:
        st.session_state.reference = text
        st.session_state.count = 1
        st.session_state.history.append((1, text))
        st.session_state.last_status = f"প্রথম উচ্চারণ শনাক্ত হয়েছে। ভাষা: {language}"
        return

    score = similarity(st.session_state.reference, text)
    threshold = st.session_state.threshold

    if score >= threshold:
        st.session_state.count += 1
        st.session_state.history.append((st.session_state.count, text))
        st.session_state.last_status = (
            f"মিল পাওয়া গেছে ({score:.0%}) • Count হয়েছে। ভাষা: {language}"
        )
        st.session_state.candidate = None
    else:
        st.session_state.candidate = text
        st.session_state.last_status = (
            f"ভিন্ন উচ্চারণ শনাক্ত হয়েছে (মিল {score:.0%})। "
            "এটি নতুন জিকির হলে নিচের বাটন ব্যবহার করুন।"
        )

# Target and matching settings
with st.expander("⚙️ সেটিংস", expanded=True):
    st.session_state.target = st.number_input(
        "Target (০ দিলে Target থাকবে না)",
        min_value=0, max_value=100000, step=10,
        value=int(st.session_state.target),
    )
    st.session_state.threshold = st.slider(
        "উচ্চারণের মিলের সীমা",
        min_value=0.50, max_value=0.95, step=0.05,
        value=float(st.session_state.threshold),
        help="সীমা কমালে কাছাকাছি বাক্যও Count হতে পারে; বেশি করলে ভুল Count কমতে পারে, তবে সঠিক উচ্চারণও বাদ পড়তে পারে।",
    )

st.markdown(f'<div class="counter">{st.session_state.count}</div>', unsafe_allow_html=True)
if st.session_state.target > 0:
    remaining = max(0, st.session_state.target - st.session_state.count)
    st.markdown(
        f'<div class="subtle">Target: {st.session_state.target} &nbsp; | &nbsp; বাকি: {remaining}</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, st.session_state.count / st.session_state.target))
else:
    st.markdown('<div class="subtle">কোনো নির্দিষ্ট Target নেই</div>', unsafe_allow_html=True)

if st.session_state.target > 0 and st.session_state.count >= st.session_state.target:
    st.success("🎉 আপনার নির্ধারিত Target পূর্ণ হয়েছে!")

st.markdown("### 🎙️ Live microphone")
st.info("Start চাপুন, মাইক্রোফোনের অনুমতি দিন, তারপর স্বাভাবিকভাবে পড়ুন। প্রতিটি অডিও অংশ নীরব বিরতির পর Speech Recognition-এ পাঠানো হবে।")

ctx = webrtc_streamer(
    key="zikr-live-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=ZikrAudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Poll audio segments from the WebRTC worker thread.
if ctx.state.playing:
    st_autorefresh(interval=1000, key="audio-poll-refresh")
    processor = ctx.audio_processor
    if processor:
        while True:
            try:
                audio = processor.audio_queue.get_nowait()
            except Exception:
                break
            try:
                with st.spinner("উচ্চারণ শনাক্ত করা হচ্ছে..."):
                    phrase, lang = transcribe(audio)
                accept_phrase(phrase, lang)
            except Exception as exc:
                st.session_state.last_status = f"Recognition error: {exc}"

st.markdown("### শনাক্ত করা উচ্চারণ")
st.write(st.session_state.last_text or "এখনো কোনো উচ্চারণ শনাক্ত হয়নি।")
st.caption(st.session_state.last_status)

if st.session_state.candidate:
    st.warning(f"নতুন/ভিন্ন উচ্চারণ: {st.session_state.candidate}")
    if st.button("এটিকে নতুন জিকির হিসেবে শুরু করুন", use_container_width=True):
        st.session_state.reference = st.session_state.candidate
        st.session_state.candidate = None
        st.session_state.count = 1
        st.session_state.history.append((1, st.session_state.reference))
        st.session_state.last_status = "নতুন জিকিরের Session শুরু হয়েছে।"
        st.rerun()

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("↩️ Undo", use_container_width=True):
        if st.session_state.history:
            st.session_state.history.pop()
            st.session_state.count = (
                st.session_state.history[-1][0]
                if st.session_state.history else 0
            )
            if st.session_state.count == 0:
                st.session_state.reference = None
            st.session_state.last_status = "সর্বশেষ Count Undo করা হয়েছে।"
            st.rerun()
with col2:
    if st.button("🆕 নতুন জিকির", use_container_width=True):
        st.session_state.count = 0
        st.session_state.reference = None
        st.session_state.last_text = ""
        st.session_state.candidate = None
        st.session_state.history = []
        st.session_state.last_status = "নতুন Session প্রস্তুত।"
        st.rerun()
with col3:
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.count = 0
        st.session_state.reference = None
        st.session_state.last_text = ""
        st.session_state.candidate = None
        st.session_state.history = []
        st.session_state.last_status = "সব Count Reset করা হয়েছে।"
        st.rerun()

with st.expander("📋 Count history"):
    if st.session_state.history:
        for number, phrase in reversed(st.session_state.history[-20:]):
            st.write(f"{number}. {phrase}")
    else:
        st.caption("এখনো কোনো Count নেই।")

st.divider()
st.caption(
    "গুরুত্বপূর্ণ: এটি একটি experimental prototype। Speech recognition ভুল করতে পারে। "
    "বিশেষ করে ছোট জিকির ও দ্রুত পুনরাবৃত্তির ক্ষেত্রে Count যাচাই করুন। "
    "মাইক্রোফোন চালু থাকলে অডিও Recognition-এর জন্য আপনার Streamlit সার্ভারে প্রক্রিয়াকরণ করা হয়।"
)
