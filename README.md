# Smart Zikr Counter

Experimental Streamlit app for live voice-based Zikr repetition counting in Arabic and Bangla pronunciation.

## Features
- Live browser microphone via `streamlit-webrtc`
- Audio segmentation after silence
- Multilingual transcription using `faster-whisper` (tiny model)
- First utterance becomes the reference phrase
- Similarity-based repetition counter
- Optional target, undo, new session, reset and recent history

## Important limitations
This is a prototype, not a validated religious counting device. Speech-to-text can omit repeated words, misrecognize Arabic/Bangla pronunciation, or merge/split utterances. Verify the displayed transcript and count. The simple energy-based audio segmenter may need tuning for your microphone and environment.

The app does not currently guarantee:
- reliable recognition of every short zikr
- fully automatic switching between different duas without user confirmation
- background counting with the phone screen off
- offline use in the hosted Streamlit version

## Run locally
Use Python 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL in Chrome and allow microphone access. For best results, use HTTPS or localhost.

## Deploy with GitHub + Streamlit Community Cloud
1. Create a GitHub repository, e.g. `smart-zikr-counter`.
2. Upload `app.py`, `audio_processor.py`, `requirements.txt`, `runtime.txt`, and `.streamlit/config.toml`.
3. Open Streamlit Community Cloud and connect the repository.
4. Select `app.py` as the entry point and deploy.
5. Open the deployed HTTPS URL on Android Chrome and allow microphone access.

## Hosting notes
- The first transcription downloads the Whisper `tiny` model and can take time.
- CPU/RAM limits on free hosting may affect latency.
- WebRTC may require STUN/TURN connectivity depending on the network. Some restricted Wi-Fi/mobile networks may block media connections.
- Audio is sent to the Streamlit server for transcription. Do not use this app for private recitations if you do not want audio processed by the server.
