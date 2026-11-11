import queue
import threading
import numpy as np
import av
from streamlit_webrtc import AudioProcessorBase

class ZikrAudioProcessor(AudioProcessorBase):
    """
    Collect mono 16 kHz audio and split utterances after a short silence.
    The resulting float32 arrays are placed in audio_queue for transcription.
    """

    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=12)
        self._resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )
        self._lock = threading.Lock()
        self._chunks = []
        self._speech_seen = False
        self._silence_samples = 0
        self._sample_rate = 16000

        # Tunable VAD-like thresholds (not a full neural VAD).
        self.energy_threshold = 450
        self.silence_duration = 0.65
        self.minimum_duration = 0.35
        self.maximum_duration = 8.0

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            resampled_frames = self._resampler.resample(frame)
            for out in resampled_frames:
                data = out.to_ndarray()
                samples = np.asarray(data).reshape(-1).astype(np.int16)
                if samples.size == 0:
                    continue

                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                with self._lock:
                    self._consume(samples, rms)
        except Exception:
            # Keep the media stream alive if one frame cannot be processed.
            pass
        return frame

    def _consume(self, samples: np.ndarray, rms: float):
        self._chunks.append(samples.copy())
        frame_duration = len(samples) / self._sample_rate

        if rms >= self.energy_threshold:
            self._speech_seen = True
            self._silence_samples = 0
        elif self._speech_seen:
            self._silence_samples += len(samples)

        total_samples = sum(len(c) for c in self._chunks)
        total_duration = total_samples / self._sample_rate
        silence_duration = self._silence_samples / self._sample_rate

        if self._speech_seen and (
            silence_duration >= self.silence_duration
            or total_duration >= self.maximum_duration
        ):
            self._emit_segment()

        # Avoid accumulating endless leading silence.
        elif not self._speech_seen and total_duration > 0.5:
            self._chunks = self._chunks[-2:]

    def _emit_segment(self):
        if not self._chunks:
            return

        audio = np.concatenate(self._chunks).astype(np.float32) / 32768.0
        duration = len(audio) / self._sample_rate

        if duration >= self.minimum_duration:
            try:
                self.audio_queue.put_nowait(audio)
            except queue.Full:
                # Drop oldest segment rather than blocking real-time audio.
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio)
                except Exception:
                    pass

        self._chunks = []
        self._speech_seen = False
        self._silence_samples = 0
