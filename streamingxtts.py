import logging
import os
import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import simpleaudio as sa
import subprocess
# Add required classes to PyTorch's safe globals
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Speaker name
SPEAKER_NAME = "Ana Florence"

class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.speaker = None

    def load(self):
        device = "cuda" 
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
            logging.info(f"Loading checkpoint from {model_path} on {device}")
            self.model.load_checkpoint(
                config, checkpoint_dir=model_path, eval=True, use_deepspeed=False
            )
            logging.info("Checkpoint loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.info("Retrying with weights_only=False")
            model_path_full = os.path.join(model_path, "model.pth")
            checkpoint = torch.load(model_path_full, map_location=torch.device(device), weights_only=False)
            try:
                self.model.load_state_dict(checkpoint["model"], strict=False)
            except RuntimeError as e:
                logging.error(f"State dict mismatch: {e}")
                state_dict = checkpoint["model"]
                new_state_dict = {k.replace("gpt.gpt_inference.", "gpt.") if "gpt.gpt_inference" in k else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
        self.model.to(device)

        self.speaker = {
            "speaker_embedding": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "speaker_embedding"
            ]
            .cpu()
            .squeeze()
            .half()
            .tolist(),
            "gpt_cond_latent": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "gpt_cond_latent"
            ]
            .cpu()
            .squeeze()
            .half()
            .tolist(),
        }

        self.speaker_embedding = (
            torch.tensor(self.speaker.get("speaker_embedding"))
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.gpt_cond_latent = (
            torch.tensor(self.speaker.get("gpt_cond_latent"))
            .reshape((-1, 1024))
            .unsqueeze(0)
        )
        logging.info("🔥 Model Loaded")

    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    def predict(self, model_input):
        text = model_input.get("text")
        language = model_input.get("language", "en")
        chunk_size = int(model_input.get("chunk_size", 60))

        streamer = self.model.inference_stream(
            text,
            language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            stream_chunk_size=chunk_size,
            enable_text_splitting=True,
            temperature=0.75,
            speed=0.9
        )

        for chunk in streamer:
            processed_chunk = self.wav_postprocess(chunk)
            processed_bytes = processed_chunk.tobytes()
            yield processed_bytes

