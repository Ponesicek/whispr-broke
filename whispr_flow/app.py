from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import pyperclip
    import sounddevice as sd
except (ImportError, ModuleNotFoundError) as exc:
    np = None
    pyperclip = None
    sd = None
    AUDIO_IMPORT_ERROR = exc
else:
    AUDIO_IMPORT_ERROR = None

try:
    from pynput import keyboard
except (ImportError, ModuleNotFoundError) as exc:
    keyboard = None
    KEYBOARD_IMPORT_ERROR = exc
else:
    KEYBOARD_IMPORT_ERROR = None


LOG = logging.getLogger("whispr-flow")
COMMANDS = {"start", "stop", "toggle", "status", "quit"}
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# v1
# TRANSCRIPTION_SYSTEM_INSTRUCTION = """
# Transcribe the attached speech audio into polished text.
#
# Rules:
# - Output only the final transcript, with no labels, commentary, Markdown, or quotes.
# - Preserve the speaker's meaning and natural wording.
# - Remove false starts, repeated stutters, filler sounds, and accidental duplicated words.
# - Correct obvious speech-recognition errors, typos, and misspellings.
# - Keep punctuation and capitalization suitable for direct insertion into a chat, document, or editor.
# - If the speaker explicitly says punctuation or formatting, apply it naturally.
# - Do not invent or infer content that is not clearly spoken in the audio.
# - If the audio is silence, music, a tone, noise, or otherwise has no clearly intelligible human speech, output nothing.
# - When user says "X emoji/emoticon", make an emoji/emoticon of X.
# """

# v2
# TRANSCRIPTION_SYSTEM_INSTRUCTION = """
# Process the provided dictation transcript into clean, naturally reading written prose.
#
# Execution Rules:
# - Translate intent: Do not transcribe disjointed speech verbatim. Rephrase awkward or rambling spoken thoughts into clear, cohesive sentences.
# - Fix errors: Automatically correct grammar, syntax, and misspoken words.
# - Clean up: Strip out all filler words, stutters, false starts, and redundancies.
# - Format properly: Apply standard punctuation and capitalization.
# - Handle commands: Execute spoken formatting (e.g., "new paragraph") and emojis (e.g., "thumbs up emoji") seamlessly.
#
# Output Constraints:
# - Output strictly the final polished text.
# - No markdown formatting, no preambles, no conversational filler.
# - If the input lacks intelligible human speech, output absolutely nothing.
# """


# v3
TRANSCRIPTION_SYSTEM_INSTRUCTION = """
Process the provided audio dictation into clean, naturally reading written prose.

Execution Rules:
- TRANSCRIBE, DO NOT CONVERSE: You are a passive dictation engine. Do not reply to greetings, answer questions, or interact with the speaker.
- Translate intent: Do not transcribe disjointed speech verbatim. Rephrase awkward or rambling spoken thoughts into clear, cohesive sentences.
- Fix errors: Automatically correct grammar, syntax, and misspoken words.
- Clean up: Strip out all filler words, stutters, false starts, and redundancies.
- Format properly: Apply standard punctuation and capitalization.
- Handle commands: Execute spoken formatting (e.g., "new paragraph") and emojis (e.g., "thumbs up emoji") seamlessly.

Output Constraints:
- Output strictly the transcribed, polished text.
- No conversational replies (e.g., do not say "I am fine"), no markdown, no preambles.
- If the audio lacks intelligible human speech, output absolutely nothing.
"""


@dataclass(frozen=True)
class Settings:
    model: str
    sample_rate: int
    channels: int
    min_seconds: float
    insert_mode: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=first_env("WHISPR_MODEL", "OPENROUTER_MODEL") or DEFAULT_MODEL,
            sample_rate=int(os.getenv("WHISPR_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("WHISPR_CHANNELS", "1")),
            min_seconds=float(os.getenv("WHISPR_MIN_SECONDS", "0.35")),
            insert_mode=os.getenv("WHISPR_INSERT_MODE", "clipboard").lower(),
        )


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


class AudioRecorder:
    def __init__(self, sample_rate: int, channels: int) -> None:
        require_audio_deps()
        self.sample_rate = sample_rate
        self.channels = channels
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._started_at = 0.0
        self._lock = threading.Lock()

    @property
    def duration(self) -> float:
        if not self._started_at:
            return 0.0
        return time.monotonic() - self._started_at

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._chunks.clear()
            self._started_at = time.monotonic()
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                callback=self._on_audio,
            )
            self._stream.start()
        LOG.info("recording started")

    def stop_to_wav(self) -> Path | None:
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is None:
            return None

        stream.stop()
        stream.close()

        with self._lock:
            chunks = list(self._chunks)
            self._chunks.clear()
            self._started_at = 0.0

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0)
        fd, name = tempfile.mkstemp(prefix="whispr-flow-", suffix=".wav")
        os.close(fd)
        path = Path(name)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio.tobytes())
        return path

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            LOG.warning("audio status: %s", status)
        with self._lock:
            self._chunks.append(indata.copy())


class Transcriber:
    def __init__(self, model: str) -> None:
        self.api_key = openrouter_api_key()
        self.model = model

    def transcribe(self, wav_path: Path) -> str:
        audio_data = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TRANSCRIPTION_SYSTEM_INSTRUCTION,
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }
        response = openrouter_request(
            OPENROUTER_URL,
            self.api_key,
            method="POST",
            payload=payload,
        )
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {response}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return clean_transcript(str(content or ""))


def openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set.")
    return api_key


def openrouter_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "whispr-flow",
    }
    referer = os.getenv("OPENROUTER_SITE_URL")
    if referer:
        headers["HTTP-Referer"] = referer
    return headers


def openrouter_request(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=openrouter_headers(api_key),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc
    return json.loads(body)


def clean_transcript(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    prefixes = ("transcript:", "transcription:", "final transcript:")
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


class TextInserter:
    def __init__(self, mode: str) -> None:
        require_audio_deps()
        if mode not in {"clipboard", "type", "print"}:
            raise ValueError("WHISPR_INSERT_MODE must be clipboard, type, or print")
        self.mode = mode

    def insert(self, text: str) -> None:
        if not text:
            LOG.info("empty transcript")
            return
        if self.mode == "print":
            print(text, flush=True)
            return
        if self.mode == "type":
            self._type_text(text)
            return
        self._paste_text(text)

    def _paste_text(self, text: str) -> None:
        pyperclip.copy(text)
        if shutil.which("xdotool"):
            subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
            return
        if shutil.which("wtype"):
            subprocess.run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"], check=False)
            return
        require_keyboard_deps()

        controller = keyboard.Controller()
        with controller.pressed(keyboard.Key.ctrl):
            controller.press("v")
            controller.release("v")

    def _type_text(self, text: str) -> None:
        if shutil.which("wtype"):
            subprocess.run(["wtype", text], check=False)
            return
        if shutil.which("xdotool"):
            subprocess.run(["xdotool", "type", "--clearmodifiers", text], check=False)
            return
        require_keyboard_deps()
        keyboard.Controller().type(text)


def play_sound(kind: str) -> None:
    if os.getenv("WHISPR_SOUND", "1").lower() in {"0", "false", "no", "off"}:
        return

    event_by_kind = {
        "start": "audio-volume-change",
        "success": "complete",
        "failure": "dialog-warning",
    }
    event = event_by_kind.get(kind, "bell")

    if shutil.which("canberra-gtk-play"):
        result = subprocess.run(
            ["canberra-gtk-play", "-i", event],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return

    sound_by_kind = {
        "success": "/usr/share/sounds/freedesktop/stereo/complete.oga",
        "failure": "/usr/share/sounds/freedesktop/stereo/dialog-warning.oga",
        "start": "/usr/share/sounds/freedesktop/stereo/audio-volume-change.oga",
    }
    sound_path = sound_by_kind.get(kind)
    if sound_path and Path(sound_path).exists():
        for player in ("paplay", "pw-play"):
            if shutil.which(player):
                result = subprocess.run(
                    [player, sound_path],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if result.returncode == 0:
                    return

    print("\a", end="", flush=True)


class DictationController:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.recorder = AudioRecorder(settings.sample_rate, settings.channels)
        self.transcriber = Transcriber(settings.model)
        self.inserter = TextInserter(settings.insert_mode)
        self.jobs: queue.Queue[Path | None] = queue.Queue()
        self.is_recording = False
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def start_recording(self) -> str:
        if self.is_recording:
            return "already-recording"
        self.is_recording = True
        try:
            self.recorder.start()
            play_sound("start")
        except Exception:
            self.is_recording = False
            LOG.exception("could not start recording")
            play_sound("failure")
            return "start-failed"
        return "recording"

    def stop_recording(self) -> str:
        if not self.is_recording:
            return "not-recording"
        self.is_recording = False
        duration = self.recorder.duration
        wav_path = self.recorder.stop_to_wav()
        if wav_path is None or duration < self.settings.min_seconds:
            if wav_path:
                wav_path.unlink(missing_ok=True)
            LOG.info("recording skipped; too short")
            return "skipped-too-short"
        self.jobs.put(wav_path)
        return "queued"

    def toggle_recording(self) -> str:
        if self.is_recording:
            return self.stop_recording()
        return self.start_recording()

    def status(self) -> str:
        return "recording" if self.is_recording else "idle"

    def shutdown(self) -> None:
        if self.is_recording:
            self.stop_recording()
        self.jobs.put(None)

    def _worker_loop(self) -> None:
        while True:
            wav_path = self.jobs.get()
            if wav_path is None:
                self.jobs.task_done()
                return
            try:
                LOG.info("transcribing %.1f KB", wav_path.stat().st_size / 1024)
                transcript = self.transcriber.transcribe(wav_path)
                self.inserter.insert(transcript)
                play_sound("success")
                LOG.info("inserted %d characters", len(transcript))
            except Exception:
                LOG.exception("transcription failed")
                play_sound("failure")
            finally:
                wav_path.unlink(missing_ok=True)
                self.jobs.task_done()


class WaylandCommandServer:
    def __init__(self, settings: Settings, socket_path: Path) -> None:
        self.controller = DictationController(settings)
        self.socket_path = socket_path
        self.should_stop = threading.Event()

    def run(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            try:
                server.bind(str(self.socket_path))
            except OSError as exc:
                LOG.warning("socket unavailable at %s: %s", self.socket_path, exc)
                LOG.info(
                    "falling back to file command directory at %s",
                    command_dir(self.socket_path),
                )
                FileCommandServer(self.controller, self.socket_path).run()
                return
            self.socket_path.chmod(0o600)
            server.listen(8)
            server.settimeout(0.5)
            LOG.info("ready: Wayland command socket at %s", self.socket_path)
            write_state(self.socket_path, self.controller.status())
            while not self.should_stop.is_set():
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    self._drain_file_commands()
                    continue
                with connection:
                    command = connection.recv(64).decode("utf-8", "replace").strip()
                    response = self._handle(command)
                    connection.sendall((response + "\n").encode("utf-8"))
        self.controller.shutdown()
        self.socket_path.unlink(missing_ok=True)

    def _handle(self, command: str) -> str:
        if command == "start":
            response = self.controller.start_recording()
            write_state(self.socket_path, self.controller.status())
            return response
        if command == "stop":
            response = self.controller.stop_recording()
            write_state(self.socket_path, self.controller.status())
            return response
        if command == "toggle":
            response = self.controller.toggle_recording()
            write_state(self.socket_path, self.controller.status())
            return response
        if command == "status":
            return self.controller.status()
        if command == "quit":
            self.should_stop.set()
            write_state(self.socket_path, "quitting")
            return "quitting"
        return f"unknown-command: {command}"

    def _drain_file_commands(self) -> None:
        for command_path in sorted(command_dir(self.socket_path).glob("*.cmd")):
            try:
                command = command_path.read_text(encoding="utf-8").strip()
                command_path.unlink(missing_ok=True)
            except OSError:
                continue
            response = self._handle(command)
            LOG.debug("file command %s -> %s", command, response)


class FileCommandServer:
    def __init__(self, controller: DictationController, socket_path: Path) -> None:
        self.controller = controller
        self.socket_path = socket_path
        self.should_stop = threading.Event()

    def run(self) -> None:
        directory = command_dir(self.socket_path)
        directory.mkdir(parents=True, exist_ok=True)
        write_state(self.socket_path, self.controller.status())
        LOG.info("ready: Wayland file command directory at %s", directory)
        while not self.should_stop.is_set():
            handled = False
            for command_path in sorted(directory.glob("*.cmd")):
                handled = True
                self._handle_file(command_path)
            if not handled:
                time.sleep(0.05)
        self.controller.shutdown()
        write_state(self.socket_path, "stopped")

    def _handle_file(self, command_path: Path) -> None:
        try:
            command = command_path.read_text(encoding="utf-8").strip()
            command_path.unlink(missing_ok=True)
        except OSError:
            return
        if command == "start":
            response = self.controller.start_recording()
        elif command == "stop":
            response = self.controller.stop_recording()
        elif command == "toggle":
            response = self.controller.toggle_recording()
        elif command == "status":
            response = self.controller.status()
        elif command == "quit":
            response = "quitting"
            self.should_stop.set()
        else:
            response = f"unknown-command: {command}"
        write_state(
            self.socket_path,
            self.controller.status() if command != "quit" else "quitting",
        )
        LOG.debug("file command %s -> %s", command, response)


class HoldHotkeyApp:
    def __init__(self, settings: Settings) -> None:
        require_keyboard_deps()
        self.controller = DictationController(settings)
        self.pressed: set[keyboard.Key | keyboard.KeyCode] = set()

    def run(self) -> None:
        LOG.info(
            "ready: hold Super+G to dictate; model=%s", self.controller.settings.model
        )
        with keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        ) as listener:
            listener.join()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        self.pressed.add(key)
        if self.controller.is_recording:
            return
        if self._is_super_g_down():
            self.controller.start_recording()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        self.pressed.discard(key)
        if not self.controller.is_recording:
            return
        if not self._is_super_g_down():
            self.controller.stop_recording()

    def _is_super_g_down(self) -> bool:
        has_super = any(
            key in self.pressed
            for key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
        )
        has_g = any(
            isinstance(key, keyboard.KeyCode) and key.char and key.char.lower() == "g"
            for key in self.pressed
        )
        return has_super and has_g


def transcribe_once(settings: Settings, wav_path: Path, print_only: bool) -> None:
    text = Transcriber(settings.model).transcribe(wav_path)
    TextInserter("print" if print_only else settings.insert_mode).insert(text)


def list_models() -> None:
    response = openrouter_request(OPENROUTER_MODELS_URL, openrouter_api_key())
    for model in response.get("data", []):
        model_id = model.get("id")
        if model_id:
            print(model_id)


def socket_path() -> Path:
    explicit = os.getenv("WHISPR_SOCKET")
    if explicit:
        return Path(explicit)
    runtime_dir = os.getenv("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return Path(runtime_dir) / "whispr-flow.sock"


def command_dir(path: Path) -> Path:
    return path.with_name(path.name + ".commands")


def state_path(path: Path) -> Path:
    return path.with_name(path.name + ".state")


def write_state(path: Path, state: str) -> None:
    try:
        state_path(path).write_text(state + "\n", encoding="utf-8")
    except OSError:
        LOG.debug("could not write state file", exc_info=True)


def enqueue_file_command(command: str, path: Path) -> str:
    directory = command_dir(path)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{time.monotonic_ns()}-{os.getpid()}.cmd"
    (directory / name).write_text(command + "\n", encoding="utf-8")
    if command == "status":
        try:
            return state_path(path).read_text(encoding="utf-8").strip() or "queued"
        except OSError:
            return "queued"
    return "queued"


def send_command(command: str, path: Path) -> str:
    if command not in COMMANDS:
        raise SystemExit(
            f"Unknown command {command!r}. Expected one of: {', '.join(sorted(COMMANDS))}"
        )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        try:
            client.connect(str(path))
        except OSError as exc:
            LOG.debug("socket command failed, using file command fallback: %s", exc)
            return enqueue_file_command(command, path)
        client.sendall(command.encode("utf-8"))
        return client.recv(1024).decode("utf-8", "replace").strip()


def require_audio_deps() -> None:
    if AUDIO_IMPORT_ERROR is None:
        return
    missing_name = getattr(AUDIO_IMPORT_ERROR, "name", None)
    if missing_name:
        detail = f"Missing desktop/audio dependency {missing_name!r}."
    else:
        detail = f"Audio backend is unavailable: {AUDIO_IMPORT_ERROR}"
    raise SystemExit(
        f"{detail} Run `uv pip install --python .venv/bin/python numpy "
        "pyperclip pynput sounddevice hatchling -e .` in this project."
    )


def require_keyboard_deps() -> None:
    if KEYBOARD_IMPORT_ERROR is None:
        return
    raise SystemExit(
        f"Global keyboard capture is unavailable: {KEYBOARD_IMPORT_ERROR}\n"
        "On Wayland, run `whispr-flow --server` and bind your compositor to "
        "`whispr-flow --command start` and `whispr-flow --command stop`."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        type=Path,
        help="Transcribe one existing WAV file instead of starting the hotkey daemon.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the transcript instead of inserting it.",
    )
    parser.add_argument(
        "--server", action="store_true", help="Run the Wayland-friendly command server."
    )
    parser.add_argument(
        "--hotkey", action="store_true", help="Run the X11 pynput Super+G listener."
    )
    parser.add_argument(
        "--command",
        choices=sorted(COMMANDS),
        help="Send a command to a running --server instance.",
    )
    parser.add_argument(
        "--socket",
        type=Path,
        default=socket_path(),
        help="Unix socket path for --server/--command.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print OpenRouter models visible to the configured API key.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    settings = Settings.from_env()

    if args.list_models:
        list_models()
        return

    if args.once:
        transcribe_once(settings, args.once, args.print_only)
        return

    if args.command:
        print(send_command(args.command, args.socket), flush=True)
        return

    if args.server or (os.getenv("WAYLAND_DISPLAY") and not args.hotkey):
        WaylandCommandServer(settings, args.socket).run()
        return

    HoldHotkeyApp(settings).run()
