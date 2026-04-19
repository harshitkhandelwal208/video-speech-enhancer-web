"""
app.py — Gradio web interface for Video Speech Enhancer.
Deployment: Render (Docker)
"""

import os
import tempfile
import torch

# ---------------------------------------------------------------------------
# Patch for Gradio 4.x + starlette 1.0 Jinja2 cache bug:
# TypeError: unhashable type: 'dict'
# Must be applied BEFORE importing gradio.
# ---------------------------------------------------------------------------
try:
    import jinja2.utils as _jinja_utils

    _OrigLRU = _jinja_utils.LRUCache

    class _PatchedLRUCache(_OrigLRU):
        def get(self, key):
            try:
                return super().get(key)
            except TypeError:
                return None

        def __getitem__(self, key):
            try:
                return super().__getitem__(key)
            except TypeError:
                raise KeyError(key)

        def __setitem__(self, key, value):
            try:
                super().__setitem__(key, value)
            except TypeError:
                pass

    _jinja_utils.LRUCache = _PatchedLRUCache
except Exception:
    pass

# ---------------------------------------------------------------------------
# Patch for Gradio 6.x bool schema bug (safety net)
# ---------------------------------------------------------------------------
try:
    import gradio_client.utils as _gc_utils

    _orig = _gc_utils._json_schema_to_python_type

    def _patched(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched
except Exception:
    pass

import gradio as gr
from audio_processor import AudioProcessor, PRESETS

# ---------------------------------------------------------------------------
# Global processor
# ---------------------------------------------------------------------------
_processor: AudioProcessor | None = None


def _get_processor() -> AudioProcessor:
    global _processor
    if _processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AudioProcessor(device=device)
    return _processor


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_video(video_path: str, preset: str, nr_strength_pct: float):
    if video_path is None:
        return None, "❌  Please upload a video file first."

    proc        = _get_processor()
    nr_strength = nr_strength_pct / 100.0
    preset_cfg  = PRESETS.get(preset, PRESETS["Balanced"])

    work_dir     = tempfile.mkdtemp()
    base         = os.path.splitext(os.path.basename(video_path))[0]
    extracted    = os.path.join(work_dir, "extracted.wav")
    enhanced     = os.path.join(work_dir, "enhanced.wav")
    static_clean = os.path.join(work_dir, "static_clean.wav")
    post_proc    = os.path.join(work_dir, "post_proc.wav")
    output_video = os.path.join(work_dir, f"cleaned_{base}.mp4")

    log: list[str] = []

    def L(msg: str):
        log.append(msg)

    try:
        L("📤  Extracting audio from video...")
        ok, msg = proc.extract_audio(video_path, extracted)
        if not ok:
            raise RuntimeError(msg)
        L(f"    ✔  {msg}")

        L(f"🧠  Running DeepFilterNet2 (NR strength {nr_strength_pct:.0f}%)...")
        ok, msg = proc.enhance_audio(
            input_audio_path=extracted,
            output_audio_path=enhanced,
            nr_strength=nr_strength,
        )
        if not ok:
            raise RuntimeError(msg)
        L(f"    ✔  {msg}")

        L("🔇  Removing residual static noise...")
        ok, msg = proc.remove_static_noise(
            enhanced,
            static_clean,
            prop_decrease=preset_cfg["static_prop_decrease"],
        )
        if not ok:
            raise RuntimeError(msg)
        L(f"    ✔  {msg}")

        L(f"🎛️   Post-processing with '{preset}' preset...")
        ok, msg = proc.post_process_audio(
            static_clean,
            post_proc,
            preset=preset,
            nr_strength_override=nr_strength,
        )
        if not ok:
            raise RuntimeError(msg)
        L(f"    ✔  {msg}")

        L("🎬  Muxing enhanced audio back into video...")
        ok, msg = proc.mux_video(video_path, post_proc, output_video)
        if not ok:
            raise RuntimeError(msg)
        L(f"    ✔  {msg}")

        L("")
        L("✅  Done!  Download your video using the player below.")
        return output_video, "\n".join(log)

    except Exception as exc:
        L(f"\n❌  Error: {exc}")
        return None, "\n".join(log)

    finally:
        for tmp in [extracted, enhanced, static_clean, post_proc]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

PRESET_DESCRIPTIONS = {
    name: cfg["description"] for name, cfg in PRESETS.items()
}


def update_description(preset: str) -> str:
    return PRESET_DESCRIPTIONS.get(preset, "")


with gr.Blocks(
    title="Video Speech Enhancer",
    css="""
        #title { font-size: 1.8rem; font-weight: 700; margin-bottom: 0; }
        #subtitle { color: #94a3b8; margin-top: 0.2rem; margin-bottom: 1.2rem; }
        #run-btn { font-size: 1.1rem; }
        .log-box textarea { font-family: monospace; font-size: 0.82rem; }
    """,
) as demo:

    gr.Markdown("# 🎙️ Video Speech Enhancer", elem_id="title")
    gr.Markdown(
        "DeepFilterNet2 noise suppression · voice-aware static removal · EQ boost",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Upload Video")

            preset = gr.Radio(
                choices=list(PRESETS.keys()),
                value="Balanced",
                label="Processing Profile",
            )

            preset_desc = gr.Markdown(
                value=PRESET_DESCRIPTIONS["Balanced"],
                elem_id="preset-desc",
            )

            nr_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=85,
                step=1,
                label="NR Strength (%)",
                info="Higher = more aggressive noise removal. 85% is a good starting point.",
            )

            run_btn = gr.Button(
                "🚀  Process Video",
                variant="primary",
                elem_id="run-btn",
            )

        with gr.Column(scale=1):
            video_out = gr.Video(label="Enhanced Output")
            log_box = gr.Textbox(
                label="Processing Log",
                lines=14,
                interactive=False,
                elem_classes=["log-box"],
            )

    gr.Markdown("---")
    gr.Markdown(
        "**Tips**\n"
        "- Works best on speech-heavy recordings (lectures, interviews, vlogs).\n"
        "- For music or ambient sound, try **Subtle** to avoid over-processing.\n"
        "- First run downloads ~30 MB of DeepFilterNet weights — subsequent runs are fast.\n"
        "- CPU processing time: roughly **1× realtime** (a 5-min video takes ~5 min)."
    )

    preset.change(fn=update_description, inputs=preset, outputs=preset_desc)

    run_btn.click(
        fn=process_video,
        inputs=[video_in, preset, nr_slider],
        outputs=[video_out, log_box],
    )


# ---------------------------------------------------------------------------
# Entry point — keep-alive ping endpoint runs on same port via Gradio's ASGI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_api=False,
        share=False,
    )
