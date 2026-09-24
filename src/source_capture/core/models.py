from __future__ import annotations

import base64
import json
import time
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        raise NotImplementedError

    def transcribe_batch(
        self, audio_paths: list[str | Path]
    ) -> list[dict[str, Any]]:
        return [self.transcribe(path) for path in audio_paths]


def create_backend(model_name: str, model_config: dict[str, Any]) -> TranscriptionBackend:
    backend = model_config.get("backend")
    if backend == "whisper_hf":
        return WhisperHFBackend(model_config)
    if backend == "qwen_http":
        return QwenHTTPBackend(model_config)
    if backend == "nemo_ctc":
        return NeMoCTCBackend(model_config)
    if backend == "ctc_triton":
        return CTCTritonBackend(model_config)
    raise ValueError(f"unsupported backend for {model_name}: {backend}")


class WhisperHFBackend(TranscriptionBackend):
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(config["model_path"])
        dtype_name = str(config.get("dtype", "float16"))
        self.dtype = getattr(torch, dtype_name)
        self.device = str(config.get("device", "cuda"))
        self.max_new_tokens = int(config.get("max_new_tokens", 180))
        self.word_timestamps = bool(config.get("word_timestamps", True))
        self.language = config.get("language")
        self.task = config.get("task")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config["model_path"], torch_dtype=self.dtype
        ).to(self.device).eval()
        from transformers import pipeline

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.dtype,
            device=self.device,
        )

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        from .audio import read_pcm16

        audio, sample_rate = read_pcm16(audio_path)
        generate_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.language:
            generate_kwargs["language"] = str(self.language)
        if self.task:
            generate_kwargs["task"] = str(self.task)
        result = self.pipeline(
            {"raw": audio, "sampling_rate": sample_rate},
            return_timestamps="word" if self.word_timestamps else None,
            generate_kwargs=generate_kwargs,
        )
        words = None
        if self.word_timestamps and result.get("chunks"):
            words = []
            for chunk in result["chunks"]:
                timestamp = chunk.get("timestamp")
                word = str(chunk.get("text", "")).strip()
                if not word or not timestamp or len(timestamp) != 2:
                    continue
                start, end = timestamp
                if start is None or end is None:
                    continue
                words.append({"word": word, "start": float(start), "end": float(end)})
        return {"hyp": str(result.get("text", "")).strip(), "lang": "", "words": words}


class QwenHTTPBackend(TranscriptionBackend):
    def __init__(self, config: dict[str, Any]):
        self.url = str(config["url"])
        self.timeout_s = float(config.get("timeout_s", 180))
        self.retries = int(config.get("retries", 3))

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        payload = json.dumps(
            {"audio": base64.b64encode(Path(audio_path).read_bytes()).decode("ascii")}
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    self.url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    value = json.loads(response.read().decode("utf-8"))
                return {
                    "hyp": value.get("text", "") or "",
                    "lang": value.get("language", "") or "",
                    "words": value.get("words"),
                }
            except Exception as exc:  # network backend must save terminal failure
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        assert last_error is not None
        raise RuntimeError(f"Qwen endpoint failed after {self.retries} attempts: {last_error}")


class NeMoCTCBackend(TranscriptionBackend):
    """Public NeMo CTC inference with greedy decoding and optional word times."""

    def __init__(self, config: dict[str, Any]):
        import torch
        from nemo.collections.asr.models import ASRModel

        self.torch = torch
        self.batch_size = int(config.get("batch_size", 1))
        self.timestamps = bool(config.get("word_timestamps", True))
        model_name = str(config.get("model_name", "nvidia/parakeet-ctc-1.1b"))
        model_path = config.get("model_path")
        if model_path and not str(model_path).startswith("__SET_ME_"):
            self.model = ASRModel.restore_from(restore_path=str(model_path))
        else:
            self.model = ASRModel.from_pretrained(model_name=model_name)
        self.model = self.model.to(str(config.get("device", "cuda"))).eval()
        window_stride = float(getattr(self.model.cfg.preprocessor, "window_stride", 0.01))
        subsampling = int(getattr(self.model.cfg.encoder, "subsampling_factor", 8))
        self.time_stride = window_stride * subsampling
        # Parakeet CTC ships with greedy decoding. Make that choice explicit so
        # a future checkpoint/config cannot silently introduce beam search or LM.
        decoding = getattr(self.model.cfg, "decoding", None)
        if decoding is not None and hasattr(self.model, "change_decoding_strategy"):
            try:
                from omegaconf import OmegaConf, open_dict

                decoding = OmegaConf.create(OmegaConf.to_container(decoding, resolve=True))
                with open_dict(decoding):
                    decoding.strategy = str(
                        config.get("decoding_strategy", "greedy_batch")
                    )
                    decoding.preserve_alignments = self.timestamps
                    decoding.compute_timestamps = self.timestamps
                self.model.change_decoding_strategy(decoding)
            except (AttributeError, TypeError, ValueError):
                # Some NeMo releases expose timestamps only as a transcribe()
                # argument. The checkpoint default remains greedy CTC.
                pass

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        return self.transcribe_batch([audio_path])[0]

    def transcribe_batch(
        self, audio_paths: list[str | Path]
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "audio": [str(path) for path in audio_paths],
            "batch_size": self.batch_size,
            "return_hypotheses": True,
            "verbose": False,
        }
        if self.timestamps:
            kwargs["timestamps"] = True
        with self.torch.inference_mode():
            try:
                outputs = self.model.transcribe(**kwargs)
            except TypeError:
                # Compatibility with NeMo versions using paths2audio_files.
                kwargs["paths2audio_files"] = kwargs.pop("audio")
                outputs = self.model.transcribe(**kwargs)
        return [self._prediction(hypothesis) for hypothesis in outputs]

    def _prediction(self, hypothesis: Any) -> dict[str, Any]:
        if isinstance(hypothesis, (tuple, list)):
            hypothesis = hypothesis[0]
        text = str(getattr(hypothesis, "text", hypothesis) or "").strip()
        words = _nemo_word_timestamps(
            getattr(hypothesis, "timestamp", None), self.time_stride
        )
        return {"hyp": text, "lang": "en", "words": words}


def _nemo_word_timestamps(
    value: Any, time_stride: float | None = None
) -> list[dict[str, Any]] | None:
    if not isinstance(value, dict):
        return None
    items = value.get("word") or value.get("words")
    if not isinstance(items, list):
        return None
    output: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word") or item.get("text") or "").strip()
        start = item.get("start")
        end = item.get("end")
        if start is None and item.get("start_offset") is not None and time_stride:
            start = float(item["start_offset"]) * time_stride
        if end is None and item.get("end_offset") is not None and time_stride:
            end = float(item["end_offset"]) * time_stride
        if not word or start is None or end is None:
            continue
        output.append({"word": word, "start": float(start), "end": float(end)})
    return output or None


class CTCTritonBackend(TranscriptionBackend):
    def __init__(self, config: dict[str, Any]):
        try:
            import tritonclient.grpc as grpcclient
        except ModuleNotFoundError as exc:
            raise RuntimeError("ctc_triton backend requires tritonclient[grpc]") from exc
        self.grpcclient = grpcclient
        self.client = grpcclient.InferenceServerClient(
            url=f"[{config['host']}]:{int(config['port'])}"
        )
        self.preprocessing_model = str(config["preprocessing_model"])
        self.ensemble_model = str(config["ensemble_model"])

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        grpcclient = self.grpcclient
        raw = np.frombuffer(Path(audio_path).read_bytes(), dtype=np.uint8)
        audio_input = grpcclient.InferInput("AUDIO_BYTES", raw.shape, "UINT8")
        audio_input.set_data_from_numpy(raw)
        pre = self.client.infer(self.preprocessing_model, [audio_input])
        error = int(pre.as_numpy("ERROR_CODE").ravel()[0])
        if error != 0:
            raise RuntimeError(f"CTC preprocessing returned ERROR_CODE={error}")
        feed = [
            ("AUDIO_WAVEFORM", "FP32"),
            ("MEL_INPUT_LENGTHS", "INT32"),
            ("POSITION_IDS", "INT32"),
            ("ERROR_CODE", "INT32"),
            ("TOTAL_DURATION", "FP32"),
            ("PADDING_MEL_FRAMES", "INT32"),
            ("ACTUAL_MEL_FRAMES", "INT32"),
        ]
        inputs = []
        for name, data_type in feed:
            value = np.ascontiguousarray(pre.as_numpy(name))
            if name == "AUDIO_WAVEFORM":
                value = value.reshape(1, 1, -1).astype(np.float32)
            elif name == "POSITION_IDS":
                value = value.reshape(1, -1).astype(np.int32)
            else:
                dtype = np.float32 if data_type == "FP32" else np.int32
                value = value.reshape(1, 1).astype(dtype)
            item = grpcclient.InferInput(name, value.shape, data_type)
            item.set_data_from_numpy(value)
            inputs.append(item)
        result = self.client.infer(self.ensemble_model, inputs)
        text = result.as_numpy("text_output")
        language = result.as_numpy("detected_language")
        return {
            "hyp": text.ravel()[0].decode() if text is not None else "",
            "lang": language.ravel()[0].decode() if language is not None else "",
            "words": None,
        }
