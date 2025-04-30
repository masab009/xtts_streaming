import logging
import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import subprocess
import io
from typing import Optional
from queue import Queue
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add required classes to PyTorch's safe globals
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI(title="TTS Streaming API")

# Queue for audio playback
playback_queue = Queue()
playback_lock = threading.Lock()

def playback_worker():
    while True:
        audio_stream, output_file = playback_queue.get()
        if audio_stream is None:  # Sentinel to stop worker
            break
        try:
            if output_file is None:
                ffplay_cmd = ["ffplay", "-nodisp", "-autoexit", '-f', 's16le', '-ar', '24000', '-ac', '1', "-"]
            else:
                logging.info(f"Saving to {output_file}")
                ffplay_cmd = ["ffmpeg", "-y", "-f", "wav", "-i", "-", output_file]

            with playback_lock:
                process = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
                try:
                    for chunk in audio_stream:
                        process.stdin.write(chunk)
                except BrokenPipeError:
                    pass
                except Exception as e:
                    logging.error(f"Playback error: {e}")
                finally:
                    process.stdin.close()
                    process.wait()
        finally:
            playback_queue.task_done()

# Start playback worker thread
playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()

class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logging.info("⏳ Downloading model")
        ModelManager().download_model(model_name)
        model_path = os.path.join(
            get_user_data_dir("tts"), model_name.replace("/", "--")
        )

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        try:
            logging.info(f"Loading checkpoint from {model_path} on {self.device}")
            self.model.load_checkpoint(
                config, checkpoint_dir=model_path, eval=True, use_deepspeed=False
            )
            logging.info("Checkpoint loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.info("Retrying with weights_only=False")
            model_path_full = os.path.join(model_path, "model.pth")
            checkpoint = torch.load(model_path_full, map_location=torch.device(self.device), weights_only=False)
            try:
                self.model.load_state_dict(checkpoint["model"], strict=False)
            except RuntimeError as e:
                logging.error(f"State dict mismatch: {e}")
                state_dict = checkpoint["model"]
                new_state_dict = {k.replace("gpt.gpt_inference.", "gpt.") if "gpt.gpt_inference" in k else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
        self.model.to(self.device)
        logging.info("🔥 Model Loaded")

    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    def get_cloned_voice_latents(self, ref_audio_path):
        """Compute conditioning latents for voice cloning from reference audio."""
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
        
        # Use get_conditioning_latents to compute latents
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=ref_audio_path,
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
            librosa_trim_db=None,
            sound_norm_refs=False,
            load_sr=22050
        )
        
        return gpt_cond_latent, speaker_embedding

    def predict(self, model_input):
        text = model_input.get("text")
        language = model_input.get("language", "en")
        chunk_size = int(model_input.get("chunk_size", 60))
        ref_audio_path = model_input.get("ref_audio_path")  # Path to reference audio

        if ref_audio_path:
            # Clone voice from reference audio
            gpt_cond_latent, speaker_embedding = self.get_cloned_voice_latents(ref_audio_path)
        else:
            # Fallback to default speaker (Ana Florence) if no reference audio is provided
            default_speaker = {
                "speaker_embedding": self.model.speaker_manager.speakers["Ana Florence"]["speaker_embedding"]
                    .cpu()
                    .squeeze()
                    .half()
                    .tolist(),
                "gpt_cond_latent": self.model.speaker_manager.speakers["Ana Florence"]["gpt_cond_latent"]
                    .cpu()
                    .squeeze()
                    .half()
                    .tolist(),
            }
            speaker_embedding = (
                torch.tensor(default_speaker.get("speaker_embedding"))
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            gpt_cond_latent = (
                torch.tensor(default_speaker.get("gpt_cond_latent"))
                .reshape((-1, 1024))
                .unsqueeze(0)
                .to(self.device)
            )

        streamer = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=chunk_size,
            enable_text_splitting=True,
            temperature=0.75,
            speed=0.9
        )

        for chunk in streamer:
            processed_chunk = self.wav_postprocess(chunk)
            processed_bytes = processed_chunk.tobytes()
            yield processed_bytes

# Initialize model
model = Model()

# Load model at startup
@app.on_event("startup")
async def startup_event():
    model.load()

class TTSInput(BaseModel):
    text: str
    language: str = "en"
    chunk_size: int = 60
    output_file: Optional[str] = None
    ref_audio_path: Optional[str] = None  # New field for reference audio

def stream_audio(audio_stream, output_file: Optional[str] = None):
    # Collect audio chunks for queuing
    chunks = []
    for chunk in audio_stream:
        chunks.append(chunk)
        yield chunk  # Stream back to client
    # Queue audio for playback
    playback_queue.put((chunks, output_file))

@app.post("/tts/stream")
async def tts_stream(input: TTSInput):
    try:
        audio_stream = model.predict(input.dict())
        return StreamingResponse(
            stream_audio(audio_stream, input.output_file),
            media_type="audio/wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing TTS: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    playback_queue.put((None, None))  # Stop playback worker

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
