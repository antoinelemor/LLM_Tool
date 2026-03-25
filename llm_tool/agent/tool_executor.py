"""
Tool executor - dispatches tool calls to existing LLMTool headless APIs.

Translates user-friendly tool arguments into the config dicts expected by
PipelineController, ModelTrainer, parallel_predict, and AnnotationValidator.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Dispatches tool calls to LLMTool's existing headless APIs."""

    def __init__(self, settings, progress_callback: Optional[Callable] = None, session_id: Optional[str] = None):
        self.settings = settings
        self.progress_callback = progress_callback
        self.session_id = session_id
        self._pipeline_controller = None

    def set_session_id(self, session_id: Optional[str]):
        """Update the session ID used for organized logging."""
        self.session_id = session_id
        self._pipeline_controller = None

    @property
    def pipeline_controller(self):
        if self._pipeline_controller is None:
            from ..pipelines.pipeline_controller import PipelineController
            self._pipeline_controller = PipelineController(
                settings=self.settings,
                progress_callback=self.progress_callback,
                session_id=self.session_id,
                session_mode="agent",
            )
        return self._pipeline_controller

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name and return results as a dict."""
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(arguments)
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return {"error": str(e), "tool": tool_name}

    # ── list_available_resources ──────────────────────────────────────────

    def _handle_list_available_resources(self, args: Dict) -> Dict:
        resource_type = args.get("resource_type", "all")
        results: Dict[str, Any] = {}

        if resource_type in ("data_files", "all"):
            results["data_files"] = self._scan_data_files()

        if resource_type in ("prompts", "all"):
            results["prompts"] = self._scan_prompts()

        if resource_type in ("trained_models", "all"):
            results["trained_models"] = self._scan_trained_models()

        if resource_type in ("ollama_models", "all"):
            results["ollama_models"] = self._list_ollama_models()

        if resource_type in ("sessions", "all"):
            results["sessions"] = self._scan_sessions()

        return results

    def _scan_data_files(self) -> List[Dict[str, str]]:
        """Scan data directory for data files."""
        data_dir = self.settings.paths.data_dir
        files = []
        if not Path(data_dir).exists():
            return files
        extensions = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}
        for p in sorted(Path(data_dir).rglob("*")):
            if p.is_file() and p.suffix.lower() in extensions:
                try:
                    size_mb = p.stat().st_size / (1024 * 1024)
                    files.append({
                        "path": str(p),
                        "name": p.name,
                        "format": p.suffix.lstrip("."),
                        "size_mb": round(size_mb, 2),
                    })
                except OSError:
                    pass
        return files

    def _scan_prompts(self) -> List[Dict[str, str]]:
        """Scan prompts directory."""
        prompts_dir = self.settings.paths.prompts_dir
        prompts = []
        if not Path(prompts_dir).exists():
            return prompts
        for p in sorted(Path(prompts_dir).rglob("*")):
            if p.is_file() and p.suffix.lower() in (".txt", ".json", ".md"):
                prompts.append({
                    "path": str(p),
                    "name": p.name,
                })
        return prompts

    def _scan_trained_models(self) -> List[Dict[str, str]]:
        """Scan models directory for trained models."""
        models_dir = self.settings.paths.models_dir
        models = []
        if not Path(models_dir).exists():
            return models
        for p in sorted(Path(models_dir).iterdir()):
            if p.is_dir():
                # Check if it contains model files
                has_model = any(
                    (p / f).exists()
                    for f in ["config.json", "model.safetensors", "pytorch_model.bin"]
                ) or any(
                    sub.is_dir() and (sub / "config.json").exists()
                    for sub in p.rglob("*")
                )
                if has_model:
                    # Try to read training metadata
                    metadata_path = p / "training_metadata.json"
                    info: Dict[str, str] = {
                        "path": str(p),
                        "name": p.name,
                    }
                    if metadata_path.exists():
                        try:
                            with open(metadata_path) as f:
                                meta = json.load(f)
                            info["accuracy"] = str(meta.get("accuracy", ""))
                            info["f1"] = str(meta.get("macro_f1", meta.get("f1", "")))
                        except Exception:
                            pass
                    models.append(info)
        return models

    def _list_ollama_models(self) -> List[str]:
        """List installed Ollama models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return [f"(ollama not available: {result.stderr.strip()})"]
            lines = result.stdout.strip().split("\n")
            models = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ["(ollama not installed or not running)"]

    def _scan_sessions(self) -> List[Dict[str, str]]:
        """Scan logs directory for resumable sessions."""
        logs_dir = self.settings.paths.logs_dir
        sessions = []
        if not Path(logs_dir).exists():
            return sessions
        for session_type in ["annotation_studio", "training_arena", "validation_lab", "data_preparation"]:
            session_dir = Path(logs_dir) / session_type
            if not session_dir.exists():
                continue
            for p in sorted(session_dir.iterdir(), reverse=True):
                if p.is_dir():
                    sessions.append({
                        "path": str(p),
                        "name": p.name,
                        "type": session_type,
                    })
                    if len(sessions) >= 20:  # Limit to recent sessions
                        break
        return sessions

    # ── preview_data ─────────────────────────────────────────────────────

    def _handle_preview_data(self, args: Dict) -> Dict:
        import pandas as pd

        file_path = args["file_path"]
        num_rows = args.get("num_rows", 5)
        path = Path(file_path)

        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(path, nrows=num_rows)
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path, nrows=num_rows)
            elif suffix == ".json":
                df = pd.read_json(path, lines=False)
                df = df.head(num_rows)
            elif suffix == ".jsonl":
                df = pd.read_json(path, lines=True, nrows=num_rows)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
                df = df.head(num_rows)
            else:
                return {"error": f"Unsupported format: {suffix}"}

            # Get full row count
            if suffix == ".csv":
                total_rows = sum(1 for _ in open(path)) - 1  # minus header
            elif suffix in (".xlsx", ".xls"):
                total_rows = len(pd.read_excel(path, usecols=[0]))
            else:
                total_rows = "unknown"

            return {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "shape": {"rows": total_rows, "columns": len(df.columns)},
                "preview": df.to_dict(orient="records"),
            }
        except Exception as e:
            return {"error": f"Failed to read {file_path}: {e}"}

    # ── run_annotation ───────────────────────────────────────────────────

    def _handle_run_annotation(self, args: Dict) -> Dict:
        config = {
            "mode": "file",
            "data_source": args["data_format"],
            "file_path": args["file_path"],
            "annotation_provider": args["annotation_provider"],
            "annotation_model": args["annotation_model"],
            "text_column": args.get("text_column", "text"),
            "prompt_path": args.get("prompt_path"),
            "batch_size": args.get("batch_size", 100),
            "max_workers": args.get("max_workers", 4),
            "run_annotation": True,
            "run_training": False,
            "run_validation": False,
            "run_deployment": False,
        }

        if args.get("api_key"):
            config["api_key"] = args["api_key"]
        if args.get("output_path"):
            config["output_path"] = args["output_path"]
        if args.get("identifier_column"):
            config["identifier_column"] = args["identifier_column"]
        if args.get("resume"):
            config["resume"] = True
            config["resume_mode"] = True

        state = self.pipeline_controller.run_pipeline(config)

        result: Dict[str, Any] = {
            "status": "success" if not state.errors else "completed_with_errors",
            "phase": state.current_phase.value,
        }
        if state.annotation_results:
            result["annotation_results"] = {
                k: v for k, v in state.annotation_results.items()
                if k in ("total_annotated", "output_file", "training_files", "training_strategy")
            }
        if state.errors:
            result["errors"] = [str(e.get("error", e)) for e in state.errors]
        if state.warnings:
            result["warnings"] = state.warnings[:10]

        return result

    # ── run_training ─────────────────────────────────────────────────────

    def _handle_run_training(self, args: Dict) -> Dict:
        from ..trainers.model_trainer import ModelTrainer

        config = {
            "input_file": args["input_file"],
            "text_column": args.get("text_column", "text"),
            "label_column": args.get("label_column", "label"),
            "model_type": args.get("model_type", "bert-base-multilingual-cased"),
            "max_epochs": args.get("max_epochs", 10),
            "batch_size": args.get("batch_size", 16),
            "learning_rate": args.get("learning_rate", 2e-5),
            "training_strategy": args.get("training_strategy", "single-label"),
            "benchmark_mode": args.get("benchmark_mode", False),
        }

        if args.get("models_to_test"):
            config["models_to_test"] = args["models_to_test"]
        if args.get("output_dir"):
            config["output_dir"] = args["output_dir"]

        trainer = ModelTrainer()
        results = trainer.train(config)

        return {
            "status": "success",
            "best_model": results.get("best_model"),
            "accuracy": results.get("accuracy"),
            "f1": results.get("f1"),
            "model_path": results.get("model_path"),
            "strategy": results.get("strategy"),
        }

    # ── run_bert_inference ───────────────────────────────────────────────

    def _handle_run_bert_inference(self, args: Dict) -> Dict:
        import pandas as pd
        import numpy as np

        model_path = args["model_path"]
        input_file = args["input_file"]
        text_column = args["text_column"]
        batch_size = args.get("batch_size", 32)

        # Read data
        path = Path(input_file)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        elif suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            return {"error": f"Unsupported format: {suffix}"}

        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found. Available: {list(df.columns)}"}

        texts = df[text_column].fillna("").tolist()

        # Run inference
        from ..trainers.parallel_inference import parallel_predict

        proba = parallel_predict(
            texts=texts,
            model_path=model_path,
            batch_size_cpu=batch_size,
            batch_size_gpu=batch_size,
            show_progress=True,
        )

        # Load label mapping
        label_map = {}
        label_map_path = Path(model_path) / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                label_map = json.load(f)
        config_path = Path(model_path) / "config.json"
        if not label_map and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            id2label = config.get("id2label", {})
            label_map = {int(k): v for k, v in id2label.items()}

        # Get predictions
        if label_map:
            predictions = [label_map.get(int(idx), str(idx)) for idx in np.argmax(proba, axis=1)]
        else:
            predictions = [int(idx) for idx in np.argmax(proba, axis=1)]

        confidences = np.max(proba, axis=1).tolist()
        df["prediction"] = predictions
        df["confidence"] = [round(c, 4) for c in confidences]

        # Save output
        output_path = args.get("output_path")
        if not output_path:
            output_path = str(path.with_stem(path.stem + "_predictions"))
        df.to_csv(output_path, index=False)

        return {
            "status": "success",
            "total_predicted": len(predictions),
            "output_file": output_path,
            "labels_found": list(set(str(p) for p in predictions)),
            "mean_confidence": round(float(np.mean(confidences)), 4),
        }

    # ── run_validation ───────────────────────────────────────────────────

    def _handle_run_validation(self, args: Dict) -> Dict:
        from ..validators.annotation_validator import AnnotationValidator

        input_file = args["input_file"]

        if not Path(input_file).exists():
            return {"error": f"File not found: {input_file}"}

        validator = AnnotationValidator()
        result = validator.validate(input_file)

        # Convert ValidationResult to dict
        return {
            "status": "success",
            "total_samples": getattr(result, "total_samples", None),
            "valid_samples": getattr(result, "valid_samples", None),
            "issues": [str(i) for i in getattr(result, "issues", [])][:20],
            "quality_score": getattr(result, "quality_score", None),
        }

    # ── run_full_pipeline ────────────────────────────────────────────────

    def _handle_run_full_pipeline(self, args: Dict) -> Dict:
        config = {
            "mode": "file",
            "data_source": args["data_format"],
            "file_path": args["file_path"],
            "annotation_provider": args["annotation_provider"],
            "annotation_model": args["annotation_model"],
            "text_column": args.get("text_column", "text"),
            "prompt_path": args.get("prompt_path"),
            "run_annotation": True,
            "run_training": True,
            "run_validation": False,
            "run_deployment": False,
            "training_model_type": args.get("training_model_type", "bert-base-multilingual-cased"),
            "training_strategy": args.get("training_strategy", "single-label"),
            "max_epochs": args.get("max_epochs", 10),
        }

        if args.get("api_key"):
            config["api_key"] = args["api_key"]

        state = self.pipeline_controller.run_pipeline(config)

        result: Dict[str, Any] = {
            "status": "success" if not state.errors else "completed_with_errors",
        }
        if state.annotation_results:
            result["annotation_results"] = {
                k: v for k, v in state.annotation_results.items()
                if k in ("total_annotated", "output_file")
            }
        if state.training_results:
            result["training_results"] = {
                k: v for k, v in state.training_results.items()
                if k in ("best_model", "accuracy", "f1", "model_path", "strategy")
            }
        if state.errors:
            result["errors"] = [str(e.get("error", e)) for e in state.errors]

        return result

    # ── transcribe_audio ──────────────────────────────────────────────

    def _handle_transcribe_audio(self, args: Dict) -> Dict:
        from ..transcription.transcriber.whisper_transcriber import WhisperTranscriber, TranscriptionConfig
        from ..transcription.transcriber.text_processor import TextProcessor
        from ..transcription.config.settings import get_config

        audio_file = args.get("audio_file") or args.get("audio_path") or args.get("file_path")
        if not audio_file:
            return {"error": "Missing required field: audio_file"}
        audio_path = Path(audio_file)
        if not audio_path.exists():
            return {"error": f"Audio file not found: {audio_file}"}

        config = get_config()
        model_name = args.get("model", "large-v3")
        language = args.get("language")
        output_format = args.get("output_format", "txt")
        diarize = args.get("diarize", False)
        output_dir = args.get("output_dir", str(self.settings.paths.data_dir / "transcripts"))

        # Create transcription config
        tc = TranscriptionConfig(
            model_name=model_name,
            language=language,
            device=config.device,
            fp16=config.use_fp16,
        )

        # Transcribe
        transcriber = WhisperTranscriber(config=tc)
        transcriber.load_model()
        result = transcriber.transcribe(audio_path)

        if not result.success:
            transcriber.cleanup()
            return {"error": f"Transcription failed: {result.error}"}

        words = result.words

        # Optional diarization
        if diarize:
            try:
                import pandas as pd
                from ..transcription.transcriber.diarization import SpeakerDiarizer, DiarizationConfig
                dc = DiarizationConfig(device=config.device)
                diarizer = SpeakerDiarizer(config=dc)
                if diarizer.load_pipeline():
                    diar_result = diarizer.diarize(audio_path)
                    if diar_result.success:
                        words_df = pd.DataFrame(words)
                        words_df = diarizer.align_words_with_speakers(words_df, diar_result)
                        words = words_df.to_dict('records')
                    diarizer.cleanup()
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")

        # Format output
        lang = result.language or 'en'
        if output_format == 'txt':
            output = TextProcessor.format_transcript(words, include_speakers=diarize, language=lang)
        elif output_format == 'csv':
            metadata = {'title': Path(audio_file).stem, 'date': '', 'video_id': '', 'channel': ''}
            output = TextProcessor.format_csv(words, metadata, language=lang)
        elif output_format == 'srt':
            output = TextProcessor.format_srt(words, language=lang)
        elif output_format == 'vtt':
            output = TextProcessor.format_vtt(words, language=lang)
        else:
            output = json.dumps({
                'text': result.text,
                'language': result.language,
                'words': words,
                'duration': result.duration,
            }, ensure_ascii=False, indent=2)

        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ext = output_format if output_format != 'vtt' else 'vtt'
        output_file = str(Path(output_dir) / f"{Path(audio_file).stem}_transcription.{ext}")
        Path(output_file).write_text(output, encoding='utf-8')

        transcriber.cleanup()

        return {
            "status": "success",
            "output_file": output_file,
            "language": result.language,
            "duration_seconds": round(result.duration, 1) if result.duration else None,
            "word_count": len(words),
            "model_used": model_name,
            "diarization": diarize,
        }

    # ── download_youtube_audio ────────────────────────────────────────

    def _handle_download_youtube_audio(self, args: Dict) -> Dict:
        from ..transcription.extractors import YouTubeExtractor
        from ..transcription.extractors.base import ExtractorConfig

        url = args.get("url") or args.get("source_url") or args.get("link")
        if not url:
            return {"error": "Missing required field: url"}
        output_dir = args.get("output_dir", str(self.settings.paths.data_dir / "audio"))
        cookies = args.get("cookies_browser")
        max_videos = args.get("max_videos", 50)
        audio_format = args.get("audio_format") or args.get("output_format") or "wav"

        ext_config = ExtractorConfig(
            output_dir=Path(output_dir),
            audio_format=audio_format,
            cookies_from_browser=cookies,
        )
        extractor = YouTubeExtractor(config=ext_config)

        # Check if it's a channel or playlist
        import re
        is_channel = bool(re.search(r'/(channel|c|@|user)/', url))
        is_playlist = 'playlist' in url.lower() or 'list=' in url

        results_list = []
        if is_channel:
            video_urls = extractor.extract_channel_videos(url, max_videos=max_videos)
            for vu in video_urls:
                r = extractor.extract_audio(vu)
                results_list.append({"url": vu, "success": r.success, "path": str(r.audio_path) if r.audio_path else None, "title": r.title, "error": r.error})
            return {"status": "success", "type": "channel", "total_videos": len(video_urls), "downloaded": sum(1 for r in results_list if r["success"]), "results": results_list[:10]}
        elif is_playlist:
            video_urls = extractor.extract_playlist(url, max_videos=max_videos)
            for vu in video_urls:
                r = extractor.extract_audio(vu)
                results_list.append({"url": vu, "success": r.success, "path": str(r.audio_path) if r.audio_path else None, "title": r.title, "error": r.error})
            return {"status": "success", "type": "playlist", "total_videos": len(video_urls), "downloaded": sum(1 for r in results_list if r["success"]), "results": results_list[:10]}
        else:
            result = extractor.extract_audio(url)
            if result.success:
                return {
                    "status": "success",
                    "type": "single_video",
                    "audio_path": str(result.audio_path),
                    "title": result.title,
                    "duration_seconds": round(result.duration, 1) if result.duration else None,
                    "video_id": result.video_id,
                }
            else:
                return {"error": f"Download failed: {result.error}"}

    # ── download_tiktok_audio ─────────────────────────────────────────

    def _handle_download_tiktok_audio(self, args: Dict) -> Dict:
        from ..transcription.extractors import TikTokExtractor
        from ..transcription.extractors.base import ExtractorConfig

        url = args.get("url") or args.get("source_url") or args.get("link")
        if not url:
            return {"error": "Missing required field: url"}
        output_dir = args.get("output_dir", str(self.settings.paths.data_dir / "audio"))
        cookies = args.get("cookies_browser")
        max_videos = args.get("max_videos", 50)
        audio_format = args.get("audio_format") or args.get("output_format") or "wav"

        ext_config = ExtractorConfig(
            output_dir=Path(output_dir),
            audio_format=audio_format,
            cookies_from_browser=cookies,
        )
        extractor = TikTokExtractor(config=ext_config)

        # Check if it's a user profile
        import re
        is_user = bool(re.search(r'/@[\w.]+/?$', url))

        if is_user:
            video_urls = extractor.extract_user_videos(url, max_videos=max_videos)
            results_list = []
            for vu in video_urls:
                r = extractor.extract_audio(vu)
                results_list.append({"url": vu, "success": r.success, "path": str(r.audio_path) if r.audio_path else None, "title": r.title, "error": r.error})
            return {"status": "success", "type": "user_profile", "total_videos": len(video_urls), "downloaded": sum(1 for r in results_list if r["success"]), "results": results_list[:10]}
        else:
            result = extractor.extract_audio(url)
            if result.success:
                return {
                    "status": "success",
                    "type": "single_video",
                    "audio_path": str(result.audio_path),
                    "title": result.title,
                    "video_id": result.video_id,
                }
            else:
                return {"error": f"Download failed: {result.error}"}

    # ── process_local_audio ───────────────────────────────────────────

    def _handle_process_local_audio(self, args: Dict) -> Dict:
        from ..transcription.extractors.local import LocalExtractor
        from ..transcription.extractors.base import ExtractorConfig

        file_path = args["file_path"]
        output_dir = args.get("output_dir", str(self.settings.paths.data_dir / "audio"))
        recursive = args.get("recursive", True)
        include_video = args.get("include_video", True)

        ext_config = ExtractorConfig(
            output_dir=Path(output_dir),
            audio_format="wav",
        )
        extractor = LocalExtractor(config=ext_config)

        path = Path(file_path)

        if path.is_dir():
            # Scan directory for audio/video files
            files = extractor.scan_directory(str(path), recursive=recursive, include_video=include_video)
            if not files:
                return {"status": "success", "message": f"No audio/video files found in {file_path}", "files_found": 0}

            results_list = []
            for f in files:
                r = extractor.extract_audio(str(f))
                results_list.append({
                    "source": str(f),
                    "success": r.success,
                    "audio_path": str(r.audio_path) if r.audio_path else None,
                    "title": r.title,
                    "error": r.error,
                })

            return {
                "status": "success",
                "type": "directory_scan",
                "files_found": len(files),
                "processed": sum(1 for r in results_list if r["success"]),
                "failed": sum(1 for r in results_list if not r["success"]),
                "results": results_list[:20],
                "output_dir": output_dir,
            }
        elif path.is_file():
            result = extractor.extract_audio(str(path))
            if result.success:
                return {
                    "status": "success",
                    "type": "single_file",
                    "audio_path": str(result.audio_path),
                    "title": result.title,
                    "source": str(path),
                }
            else:
                return {"error": f"Processing failed: {result.error}"}
        else:
            return {"error": f"Path not found: {file_path}"}

    # ── run_transcription_pipeline ────────────────────────────────────

    def _handle_run_transcription_pipeline(self, args: Dict) -> Dict:
        from ..transcription.transcriber.whisper_transcriber import WhisperTranscriber, TranscriptionConfig
        from ..transcription.transcriber.text_processor import TextProcessor
        from ..transcription.config.settings import get_config

        audio_files = args["audio_files"]
        model_name = args.get("model", "large-v3")
        language = args.get("language")
        output_format = args.get("output_format", "csv")
        diarize = args.get("diarize", False)
        output_dir = args.get("output_dir", str(self.settings.paths.data_dir / "transcripts"))

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        config = get_config()
        tc = TranscriptionConfig(
            model_name=model_name,
            language=language,
            device=config.device,
            fp16=config.use_fp16,
        )

        transcriber = WhisperTranscriber(config=tc)
        transcriber.load_model()

        # Optional diarizer
        diarizer = None
        if diarize:
            try:
                from ..transcription.transcriber.diarization import SpeakerDiarizer, DiarizationConfig
                dc = DiarizationConfig(device=config.device)
                diarizer = SpeakerDiarizer(config=dc)
                if not diarizer.load_pipeline():
                    logger.warning("Diarization pipeline could not be loaded, continuing without")
                    diarizer = None
            except Exception as e:
                logger.warning(f"Diarization unavailable: {e}")

        results_list = []
        for audio_file in audio_files:
            if not Path(audio_file).exists():
                results_list.append({"file": audio_file, "success": False, "error": "File not found"})
                continue

            result = transcriber.transcribe(Path(audio_file))
            if not result.success:
                results_list.append({"file": audio_file, "success": False, "error": result.error})
                continue

            words = result.words

            # Diarization
            if diarizer:
                try:
                    import pandas as pd
                    diar_result = diarizer.diarize(Path(audio_file))
                    if diar_result.success:
                        words_df = pd.DataFrame(words)
                        words_df = diarizer.align_words_with_speakers(words_df, diar_result)
                        words = words_df.to_dict('records')
                except Exception as e:
                    logger.warning(f"Diarization failed for {audio_file}: {e}")

            # Format output
            lang = result.language or 'en'
            if output_format == 'txt':
                output = TextProcessor.format_transcript(words, include_speakers=diarize, language=lang)
            elif output_format == 'csv':
                metadata = {'title': Path(audio_file).stem, 'date': '', 'video_id': '', 'channel': ''}
                output = TextProcessor.format_csv(words, metadata, language=lang)
            elif output_format == 'srt':
                output = TextProcessor.format_srt(words, language=lang)
            elif output_format == 'vtt':
                output = TextProcessor.format_vtt(words, language=lang)
            else:
                output = json.dumps({
                    'text': result.text,
                    'language': result.language,
                    'words': words,
                    'duration': result.duration,
                }, ensure_ascii=False, indent=2)

            ext = output_format
            output_file = str(Path(output_dir) / f"{Path(audio_file).stem}_transcription.{ext}")
            Path(output_file).write_text(output, encoding='utf-8')

            results_list.append({
                "file": audio_file,
                "success": True,
                "output_file": output_file,
                "language": result.language,
                "duration_seconds": round(result.duration, 1) if result.duration else None,
                "word_count": len(words),
            })

        transcriber.cleanup()
        if diarizer:
            diarizer.cleanup()

        return {
            "status": "success",
            "model_used": model_name,
            "total_files": len(audio_files),
            "transcribed": sum(1 for r in results_list if r.get("success")),
            "failed": sum(1 for r in results_list if not r.get("success")),
            "output_dir": output_dir,
            "output_format": output_format,
            "diarization": diarize,
            "results": results_list,
        }

    # ── tokenize_text ─────────────────────────────────────────────────

    def _handle_tokenize_text(self, args: Dict) -> Dict:
        from ..transcription.utils.text_tokenization import (
            TextTokenizer, NLPFeatures, TokenizationResult,
            format_tokenization_csv, format_tokenization_json,
        )
        from ..transcription.utils.document_loader import DocumentLoader

        input_path = args.get("input_path")
        raw_text = args.get("text")
        language = args.get("language", "en")
        seg_level = args.get("segmentation_level", 1)
        lemmatization = args.get("lemmatization", False)
        pos_tagging = args.get("pos_tagging", False)
        ner = args.get("ner", False)
        output_format = args.get("output_format", "csv")
        output_dir = args.get("output_dir", str(self.settings.paths.data_dir / "tokenization"))

        if not input_path and not raw_text:
            return {"error": "Provide either 'input_path' or 'text'"}

        features = NLPFeatures(
            lemmatization=lemmatization,
            pos_tagging=pos_tagging,
            ner=ner,
        )
        tokenizer = TextTokenizer(
            language=language,
            segmentation_level=seg_level,
            nlp_features=features,
        )

        results: List[TokenizationResult] = []

        if raw_text:
            result = tokenizer.tokenize(raw_text, document_id="text_input")
            results.append(result)
        elif input_path:
            path = Path(input_path)
            loader = DocumentLoader()

            if path.is_dir():
                loader_result = loader.load_directory(str(path))
                for doc in loader_result.documents:
                    if doc.success and doc.content:
                        doc_id = doc.title or (doc.path.stem if doc.path else "unknown")
                        result = tokenizer.tokenize(doc.content, document_id=doc_id, source_file=str(doc.path) if doc.path else None)
                        results.append(result)
            elif path.is_file():
                doc = loader.load(path)
                if not doc.success:
                    return {"error": f"Could not load document: {doc.error}"}
                result = tokenizer.tokenize(doc.content, document_id=doc.title or path.stem, source_file=str(path))
                results.append(result)
            else:
                return {"error": f"Path not found: {input_path}"}

        if not results:
            return {"error": "No text was processed"}

        # Format output
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if output_format == "csv":
            output = format_tokenization_csv(results)
            ext = "csv"
        else:
            output = format_tokenization_json(results)
            ext = "json"

        output_file = str(Path(output_dir) / f"tokenization_output.{ext}")
        Path(output_file).write_text(output, encoding='utf-8')

        # Summary stats
        total_segments = sum(len(r.segments) for r in results if r.success)
        total_sentences = sum(r.total_sentences for r in results if r.success)
        total_words = sum(r.total_words for r in results if r.success)
        total_entities = sum(r.total_entities for r in results if r.success)

        return {
            "status": "success",
            "output_file": output_file,
            "output_format": output_format,
            "documents_processed": sum(1 for r in results if r.success),
            "total_segments": total_segments,
            "total_sentences": total_sentences,
            "total_words": total_words,
            "total_entities": total_entities,
            "language": language,
            "segmentation_level": seg_level,
            "nlp_features": {
                "lemmatization": lemmatization,
                "pos_tagging": pos_tagging,
                "ner": ner,
            },
        }

    # ── load_document ─────────────────────────────────────────────────

    def _handle_load_document(self, args: Dict) -> Dict:
        from ..transcription.utils.document_loader import DocumentLoader

        file_path = args["file_path"]
        recursive = args.get("recursive", True)

        loader = DocumentLoader()
        path = Path(file_path)

        if path.is_dir():
            result = loader.load_directory(path, recursive=recursive)
            documents = []
            for doc in result.documents:
                doc_info = {
                    "path": str(doc.path) if doc.path else None,
                    "format": doc.format.value if doc.format else None,
                    "title": doc.title,
                    "success": doc.success,
                    "word_count": doc.word_count,
                    "char_count": doc.char_count,
                    "pages": doc.pages,
                }
                if doc.success and doc.content:
                    # Include first 500 chars as preview
                    doc_info["preview"] = doc.content[:500]
                if doc.error:
                    doc_info["error"] = doc.error
                documents.append(doc_info)

            return {
                "status": "success",
                "type": "directory",
                "total_files": result.total_files,
                "successful": result.successful,
                "failed": result.failed,
                "documents": documents[:50],
                "errors": result.errors[:10] if result.errors else [],
            }
        elif path.is_file():
            doc = loader.load(path)
            if not doc.success:
                return {"error": f"Failed to load document: {doc.error}"}

            return {
                "status": "success",
                "type": "single_file",
                "path": str(doc.path) if doc.path else None,
                "format": doc.format.value if doc.format else None,
                "title": doc.title,
                "author": doc.author,
                "pages": doc.pages,
                "word_count": doc.word_count,
                "char_count": doc.char_count,
                "content": doc.content,
                "metadata": doc.metadata,
            }
        else:
            return {"error": f"Path not found: {file_path}"}

    # ── Utility & Analysis Tool Handlers ────────────────────────────

    def _handle_detect_language(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detect language of text or dataset column."""
        try:
            from ..utils.language_detector import LanguageDetector
        except ImportError:
            return {"error": "Language detection not available. Install lingua-language-detector: pip install lingua-language-detector"}

        detector = LanguageDetector()
        text = args.get("text")
        file_path = args.get("file_path")

        if text:
            result = detector.detect(text, return_all_scores=True)
            return {
                "status": "success",
                "mode": "single_text",
                "language": result.get("language"),
                "language_name": detector.get_language_name(result.get("language", "")),
                "confidence": result.get("confidence"),
                "method": result.get("method"),
                "top_scores": result.get("all_scores", [])[:5],
                "recommended_models": detector.get_recommended_models(result.get("language", "en")),
            }
        elif file_path:
            import pandas as pd
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}

            text_column = args.get("text_column")
            if not text_column:
                return {"error": "text_column is required when using file_path"}

            sample_size = args.get("sample_size", 100)
            try:
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(path)
                elif path.suffix.lower() in (".xlsx", ".xls"):
                    df = pd.read_excel(path)
                elif path.suffix.lower() == ".json":
                    df = pd.read_json(path)
                elif path.suffix.lower() == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    return {"error": f"Unsupported file format: {path.suffix}"}
            except Exception as e:
                return {"error": f"Failed to load file: {e}"}

            if text_column not in df.columns:
                return {"error": f"Column '{text_column}' not found. Available: {list(df.columns)}"}

            sample = df[text_column].dropna().head(sample_size).tolist()
            results = detector.detect_batch([str(t) for t in sample])

            from collections import Counter
            lang_counts = Counter(r.get("language") for r in results)
            dominant = lang_counts.most_common(1)[0] if lang_counts else ("unknown", 0)
            avg_confidence = sum(r.get("confidence", 0) for r in results) / max(len(results), 1)

            return {
                "status": "success",
                "mode": "batch",
                "samples_analyzed": len(results),
                "dominant_language": dominant[0],
                "dominant_language_name": detector.get_language_name(dominant[0]),
                "dominant_count": dominant[1],
                "dominant_percentage": round(dominant[1] / max(len(results), 1) * 100, 1),
                "average_confidence": round(avg_confidence, 3),
                "language_distribution": dict(lang_counts.most_common(10)),
                "is_multilingual": len(lang_counts) > 1,
                "recommended_models": detector.get_recommended_models(dominant[0]),
            }
        else:
            return {"error": "Provide either 'text' for single detection or 'file_path' + 'text_column' for batch detection"}

    def _handle_estimate_annotation_cost(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate annotation cost before running a job."""
        import pandas as pd
        from ..utils.cost_estimator import resolve_model_pricing, estimate_annotation_cost, format_cost_estimate_lines

        file_path = args.get("file_path")
        text_column = args.get("text_column")
        prompt_path = args.get("prompt_path")
        provider = args.get("provider")
        model = args.get("model")
        num_samples = args.get("num_samples")

        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            return {"error": f"Prompt file not found: {prompt_path}"}

        pricing = resolve_model_pricing(provider, model)
        if pricing is None:
            return {"error": f"No pricing data for {provider}/{model}. Currently only OpenAI models have pricing data."}

        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            elif path.suffix.lower() == ".json":
                df = pd.read_json(path)
            elif path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                return {"error": f"Unsupported format: {path.suffix}"}
        except Exception as e:
            return {"error": f"Failed to load data: {e}"}

        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found. Available: {list(df.columns)}"}

        if num_samples and num_samples < len(df):
            df = df.head(num_samples)

        prompt_text = prompt_file.read_text(encoding="utf-8")
        from ..annotators.json_cleaner import extract_expected_keys
        expected_keys = extract_expected_keys(prompt_text)
        max_output_tokens = max(len(expected_keys) * 12, 100)

        prompts = [{"prompt": prompt_text, "expected_keys": expected_keys}]

        try:
            estimate = estimate_annotation_cost(
                df_subset=df,
                text_columns=[text_column],
                prompts=prompts,
                pricing=pricing,
                max_output_tokens=max_output_tokens,
            )
        except Exception as e:
            return {"error": f"Cost estimation failed: {e}"}

        lines = format_cost_estimate_lines(estimate, rich_markup=False)

        return {
            "status": "success",
            "model": estimate.model,
            "provider": estimate.provider,
            "texts_count": estimate.texts_count,
            "request_count": estimate.request_count,
            "input_tokens": estimate.input_tokens,
            "output_tokens_estimated": estimate.output_tokens_estimated,
            "input_cost_usd": round(estimate.input_cost, 4),
            "output_cost_usd": round(estimate.output_cost, 4),
            "total_cost_usd": round(estimate.total_cost, 4),
            "per_request_cost_usd": round(estimate.per_request_cost, 6),
            "currency": estimate.pricing_currency,
            "summary_lines": lines,
        }

    def _handle_analyze_dataset(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset structure intelligently."""
        from ..utils.data_detector import DataDetector

        file_path = args.get("file_path")
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        try:
            result = DataDetector.analyze_file_intelligently(path)
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}

        result["status"] = "success"
        return result

    def _handle_convert_annotations_to_training(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Convert annotation output to training-ready format."""
        from ..utils.annotation_to_training import AnnotationToTrainingConverter

        input_file = args.get("input_file")
        output_dir = args.get("output_dir")
        text_column = args.get("text_column", "sentence")
        annotation_column = args.get("annotation_column", "annotation")
        output_type = args.get("output_type", "single_label")
        label_strategy = args.get("label_strategy", "key_value")
        annotation_keys = args.get("annotation_keys")
        id_column = args.get("id_column")
        lang_column = args.get("lang_column")

        if not Path(input_file).exists():
            return {"error": f"Input file not found: {input_file}"}

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        converter = AnnotationToTrainingConverter(verbose=False)

        # First analyze the annotations
        try:
            analysis = converter.analyze_annotations(
                csv_path=input_file,
                text_column=text_column,
                annotation_column=annotation_column,
            )
        except Exception as e:
            return {"error": f"Annotation analysis failed: {e}"}

        # Then convert
        try:
            if output_type == "multi_label":
                output_path = str(Path(output_dir) / "multi_label_training.jsonl")
                result_path = converter.create_multi_label_dataset(
                    csv_path=input_file,
                    output_path=output_path,
                    text_column=text_column,
                    annotation_column=annotation_column,
                    annotation_keys=annotation_keys,
                    label_strategy=label_strategy,
                    id_column=id_column,
                    lang_column=lang_column,
                )
                return {
                    "status": "success",
                    "output_type": "multi_label",
                    "output_path": result_path,
                    "annotation_analysis": analysis,
                }
            else:
                result_paths = converter.create_single_label_datasets(
                    csv_path=input_file,
                    output_dir=output_dir,
                    text_column=text_column,
                    annotation_column=annotation_column,
                    annotation_keys=annotation_keys,
                    label_strategy=label_strategy,
                    id_column=id_column,
                    lang_column=lang_column,
                )
                return {
                    "status": "success",
                    "output_type": "single_label",
                    "output_files": result_paths,
                    "annotation_analysis": analysis,
                }
        except Exception as e:
            return {"error": f"Conversion failed: {e}"}

    def _handle_export_to_doccano(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export data to Doccano JSONL format."""
        from ..validators.doccano_exporter import DoccanoExporter

        input_file = args.get("input_file")
        output_file = args.get("output_file")
        text_column = args.get("text_column", "text")
        label_column = args.get("label_column", "label")
        sample_size = args.get("sample_size")
        stratified = args.get("stratified", True)
        confidence_threshold = args.get("confidence_threshold")

        if not Path(input_file).exists():
            return {"error": f"Input file not found: {input_file}"}

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        exporter = DoccanoExporter()

        try:
            if sample_size:
                result = exporter.create_validation_sample(
                    annotated_file=input_file,
                    output_file=output_file,
                    sample_size=sample_size,
                    stratified=stratified,
                    text_column=text_column,
                    label_column=label_column,
                )
            else:
                result = exporter.export_to_doccano(
                    data=input_file,
                    output_path=output_file,
                    text_column=text_column,
                    label_column=label_column,
                    include_metadata=True,
                    confidence_threshold=confidence_threshold,
                )
        except Exception as e:
            return {"error": f"Export failed: {e}"}

        result["status"] = "success"
        return result

    def _handle_detect_system_resources(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detect system resources and return optimization recommendations."""
        from ..utils.system_resources import detect_resources, get_device_recommendation, get_optimal_workers, get_optimal_batch_size, get_model_optimized_config

        try:
            resources = detect_resources(force_refresh=True)
        except Exception as e:
            return {"error": f"Resource detection failed: {e}"}

        result = {
            "status": "success",
            "gpu": {
                "available": resources.gpu.available,
                "device_type": resources.gpu.device_type,
                "device_count": resources.gpu.device_count,
                "device_names": resources.gpu.device_names,
                "total_memory_gb": resources.gpu.total_memory_gb,
                "available_memory_gb": resources.gpu.available_memory_gb,
            },
            "cpu": {
                "physical_cores": resources.cpu.physical_cores,
                "logical_cores": resources.cpu.logical_cores,
                "processor_name": resources.cpu.processor_name,
            },
            "memory": {
                "total_gb": resources.memory.total_gb,
                "available_gb": resources.memory.available_gb,
                "percent_used": resources.memory.percent_used,
            },
            "storage": {
                "total_gb": resources.storage.total_gb,
                "available_gb": resources.storage.available_gb,
            },
            "recommendations": {
                "device": get_device_recommendation(),
                "optimal_workers": get_optimal_workers(),
                "optimal_batch_size": get_optimal_batch_size(),
            },
        }

        model_name = args.get("model_name")
        if model_name:
            try:
                model_config = get_model_optimized_config(model_name, force_refresh=True)
                result["model_config"] = model_config
            except Exception as e:
                result["model_config_error"] = str(e)

        return result

    def _handle_split_dataset(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Split dataset into train/val/test sets."""
        import pandas as pd

        input_file = args.get("input_file")
        output_dir = args.get("output_dir")
        text_column = args.get("text_column", "text")
        label_column = args.get("label_column", "label")
        train_ratio = args.get("train_ratio", 0.8)
        val_ratio = args.get("val_ratio", 0.1)
        test_ratio = args.get("test_ratio", 0.1)
        stratify = args.get("stratify", True)
        random_seed = args.get("random_seed", 42)

        path = Path(input_file)
        if not path.exists():
            return {"error": f"File not found: {input_file}"}

        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            elif path.suffix.lower() == ".json":
                df = pd.read_json(path)
            elif path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                return {"error": f"Unsupported format: {path.suffix}"}
        except Exception as e:
            return {"error": f"Failed to load file: {e}"}

        if text_column not in df.columns:
            return {"error": f"Text column '{text_column}' not found. Available: {list(df.columns)}"}

        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            return {"error": "scikit-learn required for dataset splitting. Install: pip install scikit-learn"}

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stratify_col = None
        if stratify and label_column in df.columns:
            # Only stratify if we have enough samples per class
            min_class_count = df[label_column].value_counts().min()
            if min_class_count >= 3:
                stratify_col = df[label_column]

        try:
            # First split: train vs (val+test)
            val_test_ratio = val_ratio + test_ratio
            train_df, temp_df = train_test_split(
                df, test_size=val_test_ratio, random_state=random_seed,
                stratify=stratify_col,
            )

            # Second split: val vs test
            temp_stratify = None
            if stratify_col is not None and label_column in temp_df.columns:
                min_temp = temp_df[label_column].value_counts().min()
                if min_temp >= 2:
                    temp_stratify = temp_df[label_column]

            relative_test = test_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df, test_size=relative_test, random_state=random_seed,
                stratify=temp_stratify,
            )
        except Exception as e:
            return {"error": f"Split failed: {e}"}

        # Save
        train_path = out / "train.csv"
        val_path = out / "val.csv"
        test_path = out / "test.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        result = {
            "status": "success",
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "stratified": stratify_col is not None,
            "random_seed": random_seed,
        }

        if label_column in df.columns:
            result["label_distribution"] = {
                "train": dict(train_df[label_column].value_counts().head(10)),
                "val": dict(val_df[label_column].value_counts().head(10)),
                "test": dict(test_df[label_column].value_counts().head(10)),
            }

        return result

    def _handle_manage_prompts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List, read, or analyze prompt files."""
        action = args.get("action")
        prompt_path = args.get("prompt_path")

        prompts_dir = getattr(self.settings.paths, "prompts_dir", Path("prompts"))
        override_dir = args.get("directory") or args.get("prompts_dir")
        prompts_dir = Path(override_dir) if override_dir else Path(prompts_dir)

        if action == "list":
            if not prompts_dir.exists():
                return {"status": "success", "prompts": [], "message": f"Prompts directory not found: {prompts_dir}"}

            prompt_files = []
            for ext in ("*.txt", "*.md", "*.prompt"):
                prompt_files.extend(prompts_dir.glob(ext))
            prompt_files.sort()

            prompts_info = []
            for pf in prompt_files:
                try:
                    content = pf.read_text(encoding="utf-8")
                    prompts_info.append({
                        "name": pf.name,
                        "path": str(pf),
                        "size_bytes": pf.stat().st_size,
                        "lines": content.count("\n") + 1,
                        "preview": content[:200].strip(),
                    })
                except Exception:
                    prompts_info.append({"name": pf.name, "path": str(pf), "error": "Could not read"})

            return {"status": "success", "count": len(prompts_info), "prompts": prompts_info}

        elif action == "read":
            if not prompt_path:
                return {"error": "prompt_path is required for 'read' action"}
            path = Path(prompt_path)
            if not path.exists():
                return {"error": f"Prompt file not found: {prompt_path}"}
            try:
                content = path.read_text(encoding="utf-8")
                return {
                    "status": "success",
                    "name": path.name,
                    "path": str(path),
                    "content": content,
                    "lines": content.count("\n") + 1,
                    "size_bytes": path.stat().st_size,
                }
            except Exception as e:
                return {"error": f"Failed to read prompt: {e}"}

        elif action == "analyze":
            if not prompt_path:
                return {"error": "prompt_path is required for 'analyze' action"}
            path = Path(prompt_path)
            if not path.exists():
                return {"error": f"Prompt file not found: {prompt_path}"}
            try:
                content = path.read_text(encoding="utf-8")
                from ..annotators.json_cleaner import extract_expected_keys
                expected_keys = extract_expected_keys(content)
                return {
                    "status": "success",
                    "name": path.name,
                    "path": str(path),
                    "expected_keys": expected_keys,
                    "key_count": len(expected_keys),
                    "preview": content[:300].strip(),
                }
            except Exception as e:
                return {"error": f"Prompt analysis failed: {e}"}
        else:
            return {"error": f"Unknown action: {action}. Use 'list', 'read', or 'analyze'."}
