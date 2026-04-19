"""
audio_processor.py
Backend for Video Speech Enhancer — no GUI dependencies.
Shared between the desktop Tkinter app and the Gradio web app.
"""

import os
import sys
import subprocess
from typing import Optional, Tuple, Callable

import numpy as np
import soundfile as sf
import imageio_ffmpeg
import torch
import torchaudio  # noqa: F401  (imported for any torchaudio ops if needed)


# ---------------------------------------------------------------------------
# Processing profile presets
# ---------------------------------------------------------------------------
PRESETS = {
    "Subtle": {
        "description": "Light touch — minimal artefacts, preserves naturalness",
        "hp_cutoff_hz": 60,
        "nr_prop_decrease": 0.30,
        "static_prop_decrease": 0.40,
        "eq_2k_db": 2.0,
        "eq_3k5_db": 1.0,
    },
    "Balanced": {
        "description": "General-purpose — good noise removal with clean speech",
        "hp_cutoff_hz": 80,
        "nr_prop_decrease": 0.50,
        "static_prop_decrease": 0.55,
        "eq_2k_db": 4.0,
        "eq_3k5_db": 2.5,
    },
    "Aggressive": {
        "description": "Maximum noise removal — best for very noisy environments",
        "hp_cutoff_hz": 100,
        "nr_prop_decrease": 0.70,
        "static_prop_decrease": 0.75,
        "eq_2k_db": 6.0,
        "eq_3k5_db": 4.0,
    },
    "Podcast": {
        "description": "Optimised for voice-only content — warmth + presence",
        "hp_cutoff_hz": 80,
        "nr_prop_decrease": 0.45,
        "static_prop_decrease": 0.50,
        "eq_2k_db": 3.5,
        "eq_3k5_db": 4.0,
    },
}


class AudioProcessor:
    """Backend for audio extraction, enhancement, and muxing operations."""

    # DeepFilterNet2 – real-time speech enhancement (NOT separation).
    # Works at 48 kHz natively; far superior to SepFormer for single-speaker
    # noise-suppression (no musical-noise / chopping artefacts).
    MODEL_NAME = "DeepFilterNet2"
    TARGET_SAMPLE_RATE = 48000  # DeepFilterNet native sample rate

    def __init__(
        self,
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.progress_callback = progress_callback
        self.model = None
        self.df_state = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        def report(stage: str, percent: float):
            if self.progress_callback:
                self.progress_callback(stage, percent)

        report("Checking for DeepFilterNet...", 10)

        # Fix for environments where stderr/stdout may be None
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")

        # Auto-install deepfilternet if missing
        try:
            from df import init_df  # noqa: F401
        except ImportError:
            report("Installing DeepFilterNet (first run, ~30 MB)...", 15)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "deepfilternet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        report("Loading DeepFilterNet2 model weights...", 40)
        from df import init_df
        # init_df() downloads weights on first run (~30 MB), then caches them.
        # Device is handled internally by DeepFilterNet.
        self.model, self.df_state, _ = init_df()

        report("Model ready", 100)

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    def extract_audio(self, video_path: str, output_audio_path: str) -> Tuple[bool, str]:
        """Extract mono 48 kHz WAV from video using ffmpeg."""
        try:
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            if not os.path.exists(ffmpeg_bin):
                return False, "FFmpeg binary not found"

            cmd = [
                ffmpeg_bin,
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(self.TARGET_SAMPLE_RATE),
                "-y",
                output_audio_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            if result.returncode != 0:
                return False, f"FFmpeg extraction failed: {result.stderr}"

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
                return False, "Extracted audio file is empty or missing"

            return True, "Audio extracted successfully"

        except Exception as e:
            return False, f"Audio extraction error: {str(e)}"

    # ------------------------------------------------------------------
    # AI enhancement (DeepFilterNet2)
    # ------------------------------------------------------------------

    def enhance_audio(
        self,
        input_audio_path: str,
        output_audio_path: str,
        progress_callback=None,
        nr_strength: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Enhance audio using DeepFilterNet2.

        DeepFilterNet2 is a causal, real-time speech enhancement model trained
        on the DNS and VoiceBank+DEMAND datasets. Unlike SepFormer (which
        performs *speech separation* and chops up single-speaker audio),
        DeepFilterNet suppresses noise while keeping the voice completely
        intact — no musical noise, no chopping artefacts.

        Parameters
        ----------
        nr_strength : float, 0.0 – 1.0
            Maps to DeepFilterNet's ``atten_lim_db`` parameter:
              0.0  →  no suppression (pass-through)
              0.5  →  moderate (~50 dB attenuation limit)
              1.0  →  unlimited / maximum suppression
        """
        try:
            from df.enhance import enhance
            from df.io import load_audio, save_audio

            if progress_callback:
                progress_callback("Loading audio into DeepFilterNet...")

            # load_audio resamples to df_state.sr() (48 kHz) automatically
            audio, _ = load_audio(input_audio_path, sr=self.df_state.sr())

            if progress_callback:
                progress_callback(
                    f"Running DeepFilterNet2 speech enhancement "
                    f"(NR strength {nr_strength:.0%})..."
                )

            # atten_lim_db: None = unlimited (max suppression);
            # a finite value caps how many dB of noise gets removed.
            if nr_strength >= 0.99:
                atten_lim = None          # full power
            elif nr_strength <= 0.01:
                atten_lim = 0.0           # pass-through
            else:
                atten_lim = nr_strength * 100.0  # 0 – 100 dB range

            with torch.no_grad():
                enhanced_audio = enhance(
                    self.model,
                    self.df_state,
                    audio,
                    atten_lim_db=atten_lim,
                )

            if progress_callback:
                progress_callback("Saving DeepFilterNet-enhanced audio...")

            save_audio(output_audio_path, enhanced_audio, self.df_state.sr())

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) < 100:
                size = os.path.getsize(output_audio_path) if os.path.exists(output_audio_path) else 0
                return False, f"Enhanced audio file not created properly (size: {size} bytes)"

            return True, "DeepFilterNet2 enhancement completed"

        except torch.cuda.OutOfMemoryError:
            return False, "GPU out of memory. Try a shorter video or switch to CPU."
        except Exception as e:
            return False, f"Enhancement error: {str(e)}"

    # ------------------------------------------------------------------
    # Static noise removal — voice-aware
    # ------------------------------------------------------------------

    def remove_static_noise(
        self,
        input_audio_path: str,
        output_audio_path: str,
        prop_decrease: float = 0.90,
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """
        Voice-aware static noise removal using quietest segment as noise profile.

        Strategy
        --------
        1. Scan entire audio to find quietest 500ms → use as noise fingerprint.
        2. Detect voiced frames using energy + spectral centroid VAD.
        3. Apply gentle spectral subtraction ONLY to non-speech frames.
        4. Crossfade at boundaries (20ms ramp) for smooth transitions.
        5. RMS-match output to input to preserve perceived loudness.
        6. Peak-normalise to 0.95 FS.
        """
        try:
            import noisereduce as nr
            from scipy.signal import butter, sosfilt  # noqa: F401

            if progress_callback:
                progress_callback("Loading audio for voice-aware static noise removal...")

            audio_data, sample_rate = sf.read(input_audio_path, dtype="float32")

            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1).astype(np.float32)

            total_samples = len(audio_data)

            # Step 1: Find quietest 500ms window → noise fingerprint
            if progress_callback:
                progress_callback("Analyzing audio to find quietest segment (noise profile)...")

            win_samples = int(0.50 * sample_rate)
            step = max(1, win_samples // 4)
            best_rms = float("inf")
            best_start = 0
            noise_clip = audio_data[:win_samples] if total_samples >= win_samples else audio_data.copy()

            for start in range(0, max(1, total_samples - win_samples), step):
                seg = audio_data[start: start + win_samples]
                rms = float(np.sqrt(np.mean(seg ** 2)))
                if rms < best_rms:
                    best_rms = rms
                    noise_clip = seg.copy()
                    best_start = start

            noise_floor_rms = best_rms
            noise_floor_db = 20 * np.log10(noise_floor_rms + 1e-10)

            if progress_callback:
                progress_callback(
                    f"Noise floor: {noise_floor_db:.1f} dB (at {best_start / sample_rate:.2f}s)"
                )

            # Step 2: Energy-based Voice Activity Detection (VAD)
            if progress_callback:
                progress_callback("Running voice activity detection (VAD)...")

            frame_ms = 20
            frame_size = int(frame_ms * sample_rate / 1000)
            snr_threshold_db = 8.0
            snr_linear = 10 ** (snr_threshold_db / 20.0)
            speech_rms_threshold = noise_floor_rms * snr_linear

            n_frames = max(1, total_samples // frame_size)
            is_speech_frame = np.zeros(n_frames, dtype=bool)

            fft_freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
            voiced_lo, voiced_hi = 200.0, 4000.0

            for i in range(n_frames):
                s = i * frame_size
                e = min(s + frame_size, total_samples)
                frame = audio_data[s:e]
                frame_rms = float(np.sqrt(np.mean(frame ** 2)))
                if frame_rms < speech_rms_threshold:
                    continue
                mag = np.abs(np.fft.rfft(frame))
                total_mag = mag.sum()
                if total_mag < 1e-10:
                    continue
                centroid = float(np.dot(fft_freqs, mag) / total_mag)
                if voiced_lo <= centroid <= voiced_hi:
                    is_speech_frame[i] = True

            # Dilate ±60ms so we don't clip word onset/offset
            dilation_frames = max(1, int(60 / frame_ms))
            dilated = is_speech_frame.copy()
            for offset in range(1, dilation_frames + 1):
                dilated[offset:] |= is_speech_frame[:-offset]
                dilated[:-offset] |= is_speech_frame[offset:]
            is_speech_frame = dilated

            speech_pct = 100.0 * is_speech_frame.sum() / n_frames
            if progress_callback:
                progress_callback(
                    f"VAD: {speech_pct:.0f}% of frames detected as speech "
                    f"(those frames will NOT be filtered)"
                )

            # Step 3: Full spectral subtraction pass
            if progress_callback:
                progress_callback(
                    f"Applying spectral subtraction to non-speech frames "
                    f"(strength {prop_decrease:.0%})..."
                )

            denoised_full = nr.reduce_noise(
                y=audio_data,
                y_noise=noise_clip,
                sr=sample_rate,
                stationary=True,
                prop_decrease=prop_decrease,
                freq_mask_smooth_hz=300,
                time_mask_smooth_ms=40,
            ).astype(np.float32)

            # Step 4: Blend — speech frames keep original, silence gets denoised
            if progress_callback:
                progress_callback("Blending: restoring original voice on speech frames...")

            weight = np.zeros(total_samples, dtype=np.float32)
            for i in range(n_frames):
                if is_speech_frame[i]:
                    s = i * frame_size
                    e = min(s + frame_size, total_samples)
                    weight[s:e] = 1.0

            fade_samples = int(0.020 * sample_rate)
            if fade_samples > 1:
                kernel = np.hanning(fade_samples * 2 + 1).astype(np.float32)
                kernel /= kernel.sum()
                weight = np.convolve(weight, kernel, mode="same")
                weight = np.clip(weight, 0.0, 1.0)

            output = weight * audio_data + (1.0 - weight) * denoised_full

            # Step 5: RMS-match to preserve perceived loudness
            in_rms  = float(np.sqrt(np.mean(audio_data ** 2))) + 1e-9
            out_rms = float(np.sqrt(np.mean(output ** 2)))      + 1e-9
            output  = (output * (in_rms / out_rms)).astype(np.float32)

            # Step 6: Peak-normalise to 0.95 FS
            max_val = np.max(np.abs(output))
            if max_val > 0 and max_val < 1e-6:
                return False, f"Static-removed audio is silent (max: {max_val})"
            if max_val > 0:
                output = output / max_val * 0.95

            if progress_callback:
                progress_callback("Saving voice-protected static-removed audio...")

            sf.write(output_audio_path, output, sample_rate, subtype="PCM_16")

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) < 100:
                return False, "Static-removed audio not created properly"

            return True, "Voice-aware static noise removal completed"

        except ImportError as e:
            return False, f"Missing dependency: {e}. Run: pip install noisereduce scipy"
        except Exception as e:
            return False, f"Static noise removal error: {str(e)}"

    # ------------------------------------------------------------------
    # Post-processing  (high-pass + spectral gate + EQ)
    # ------------------------------------------------------------------

    def post_process_audio(
        self,
        input_audio_path: str,
        output_audio_path: str,
        preset: str = "Balanced",
        nr_strength_override: Optional[float] = None,
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """
        Apply spectral noise gating and peaking EQ.

        Pipeline:
            1. High-pass filter (cutoff from preset) — kills sub-bass rumble.
            2. noisereduce non-stationary spectral gate (strength from preset or slider).
            3. Peaking EQ +N dB @ 2kHz   — consonant intelligibility.
            4. Peaking EQ +N dB @ 3.5kHz — speech air and clarity.
            5. Peak-normalise to 0.95.
        """
        try:
            import noisereduce as nr
            from scipy.signal import butter, sosfilt, lfilter

            cfg = PRESETS.get(preset, PRESETS["Balanced"])
            hp_cutoff = cfg["hp_cutoff_hz"]
            nr_prop   = nr_strength_override if nr_strength_override is not None else cfg["nr_prop_decrease"]
            eq_2k_db  = cfg["eq_2k_db"]
            eq_3k5_db = cfg["eq_3k5_db"]

            if progress_callback:
                progress_callback(
                    f"Post-processing with preset '{preset}' (NR strength {nr_prop:.0%})..."
                )

            audio_data, sample_rate = sf.read(input_audio_path, dtype="float32")

            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1).astype(np.float32)

            # ── Step 1: High-pass filter ──────────────────────────────
            if progress_callback:
                progress_callback(f"Applying {hp_cutoff}Hz high-pass filter...")

            sos_hp = butter(4, float(hp_cutoff), btype="highpass", fs=sample_rate, output="sos")
            audio_data = sosfilt(sos_hp, audio_data).astype(np.float32)

            # ── Step 2: Voice-aware spectral gate ─────────────────────
            if progress_callback:
                progress_callback("Running voice-aware spectral noise gate...")

            win_s  = int(0.30 * sample_rate)
            step_s = max(1, win_s // 2)
            best_rms_s = float("inf")
            noise_clip = audio_data[:win_s] if len(audio_data) >= win_s else audio_data.copy()
            for st in range(0, max(1, len(audio_data) - win_s), step_s):
                seg = audio_data[st: st + win_s]
                r = float(np.sqrt(np.mean(seg ** 2)))
                if r < best_rms_s:
                    best_rms_s = r
                    noise_clip = seg.copy()

            frame_sz    = int(0.020 * sample_rate)
            snr_thr     = best_rms_s * (10 ** (8.0 / 20.0))
            fft_f       = np.fft.rfftfreq(frame_sz, d=1.0 / sample_rate)
            n_fr        = max(1, len(audio_data) // frame_sz)
            speech_mask = np.zeros(n_fr, dtype=bool)
            for i in range(n_fr):
                s, e = i * frame_sz, min((i + 1) * frame_sz, len(audio_data))
                fr = audio_data[s:e]
                if float(np.sqrt(np.mean(fr ** 2))) >= snr_thr:
                    mag = np.abs(np.fft.rfft(fr))
                    tot = mag.sum()
                    if tot > 1e-10:
                        c = float(np.dot(fft_f, mag) / tot)
                        if 200.0 <= c <= 4000.0:
                            speech_mask[i] = True

            dil = max(1, int(60 / 20))
            dilated_s = speech_mask.copy()
            for off in range(1, dil + 1):
                dilated_s[off:]  |= speech_mask[:-off]
                dilated_s[:-off] |= speech_mask[off:]
            speech_mask = dilated_s

            denoised_pp = nr.reduce_noise(
                y=audio_data,
                y_noise=noise_clip,
                sr=sample_rate,
                stationary=False,
                prop_decrease=nr_prop,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
            ).astype(np.float32)

            weight_s = np.zeros(len(audio_data), dtype=np.float32)
            for i in range(n_fr):
                if speech_mask[i]:
                    s, e = i * frame_sz, min((i + 1) * frame_sz, len(audio_data))
                    weight_s[s:e] = 1.0
            fade_s = int(0.020 * sample_rate)
            if fade_s > 1:
                kern = np.hanning(fade_s * 2 + 1).astype(np.float32)
                kern /= kern.sum()
                weight_s = np.clip(np.convolve(weight_s, kern, mode="same"), 0.0, 1.0)

            audio_data = (weight_s * audio_data + (1.0 - weight_s) * denoised_pp).astype(np.float32)

            # RMS-match to prevent level shift from blending
            pre_rms   = float(np.sqrt(np.mean(audio_data ** 2))) + 1e-9
            blend_rms = float(np.sqrt(np.mean(audio_data ** 2))) + 1e-9
            if blend_rms > 1e-9:
                audio_data = (audio_data * (pre_rms / blend_rms)).astype(np.float32)

            # ── Steps 3 & 4: Peaking EQ ───────────────────────────────
            def apply_peaking_eq(
                signal: np.ndarray, center_hz: float, gain_db: float, Q: float, fs: int
            ) -> np.ndarray:
                w0    = 2.0 * np.pi * center_hz / fs
                alpha = np.sin(w0) / (2.0 * Q)
                A     = 10.0 ** (gain_db / 40.0)
                b0, b1, b2 = 1.0 + alpha * A, -2.0 * np.cos(w0), 1.0 - alpha * A
                a0, a1, a2 = 1.0 + alpha / A, -2.0 * np.cos(w0), 1.0 - alpha / A
                b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
                a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
                return lfilter(b, a, signal.astype(np.float64)).astype(np.float32)

            if progress_callback:
                progress_callback(
                    f"Applying EQ +{eq_2k_db}dB @ 2kHz, +{eq_3k5_db}dB @ 3.5kHz..."
                )

            audio_data = apply_peaking_eq(audio_data, 2000.0, eq_2k_db,  1.5, sample_rate)
            audio_data = apply_peaking_eq(audio_data, 3500.0, eq_3k5_db, 2.0, sample_rate)

            # ── Step 5: Normalise ─────────────────────────────────────
            if progress_callback:
                progress_callback("Normalising post-processed audio...")

            max_val = np.max(np.abs(audio_data))
            if max_val > 0 and max_val < 1e-6:
                return False, f"Post-processed audio is silent (max: {max_val})"
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95

            if progress_callback:
                progress_callback("Saving post-processed audio...")

            sf.write(output_audio_path, audio_data, sample_rate, subtype="PCM_16")

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) < 100:
                return False, "Post-processed audio not created properly"

            return True, "Post-processing completed"

        except ImportError as e:
            return False, f"Missing dependency: {e}. Run: pip install noisereduce scipy"
        except Exception as e:
            return False, f"Post-processing error: {str(e)}"

    # ------------------------------------------------------------------
    # Muxing
    # ------------------------------------------------------------------

    def mux_video(
        self,
        original_video_path: str,
        enhanced_audio_path: str,
        output_video_path: str,
    ) -> Tuple[bool, str]:
        """Mux enhanced audio back into original video without re-encoding video."""
        try:
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

            if not os.path.exists(enhanced_audio_path):
                return False, f"Enhanced audio file not found: {enhanced_audio_path}"

            audio_file_size = os.path.getsize(enhanced_audio_path)
            if audio_file_size < 100:
                return False, f"Enhanced audio file too small: {audio_file_size} bytes"

            cmd = [
                ffmpeg_bin,
                "-i", original_video_path,
                "-i", enhanced_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-y",
                output_video_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            if result.returncode != 0:
                return False, f"FFmpeg muxing failed: {result.stderr}"

            if not os.path.exists(output_video_path):
                return False, "Output video file was not created"

            output_size = os.path.getsize(output_video_path)
            if output_size < 1000:
                return False, f"Output video too small: {output_size} bytes (possible corruption)"

            return True, "Video muxing completed successfully"

        except Exception as e:
            return False, f"Video muxing error: {str(e)}"