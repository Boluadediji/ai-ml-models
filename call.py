# voice_rag_agent.py
"""
Realtime voice RAG agent for transfer & card issues.

How to use:
- Replace `receive_audio_callback` and `send_audio_callback` with your RTC/SignalR code.
- Provide `get_azure_client`, `classify_intent`, and `retrieve_relevant_doc` as in your stack.
- Set AZURE_SPEECH_KEY / AZURE_SPEECH_REGION in env.
"""

import asyncio
import os
import json
import traceback
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import sounddevice as sd
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# ---------- Your imports (adapt these modules) ----------
# from intent_classifier import classify_intent
# from ragengine import retrieve_relevant_doc
# from config import get_azure_client

# For this example we create small stubs so file is runnable for testing
def classify_intent(text: str) -> str:
    # Replace with your classifier
    t = text.lower()
    if "card" in t:
        return "card_issue"
    if "transfer" in t or "payment" in t:
        return "transfer_issue"
    return "unsupported_request"

def retrieve_relevant_doc(intent: str, text: str) -> Tuple[Optional[str], float]:
    # Replace with your RAG engine call. Return (doc_text, score)
    if intent == "card_issue":
        return ("Policy: Card - steps to block/suspend & dispute.", 0.92)
    if intent == "transfer_issue":
        return ("Policy: Transfers - failed/stuck/incorrect steps.", 0.88)
    return (None, 0.0)

def get_azure_client():
    # Replace with your Azure-OpenAI wrapper client that supports client.chat.completions.create
    raise NotImplementedError("Replace get_azure_client() with your Azure client builder")

# ---------- Config ----------
load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")
if not (AZURE_SPEECH_KEY and AZURE_REGION):
    raise RuntimeError("Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in your env")

# ---------- Azure Speech SDK setup ----------
speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
speech_config.speech_recognition_language = "en-NG"
speech_config.speech_synthesis_voice_name = "en-NG-AdaNeural"

# ---------- Conversation & state helpers ----------
MAX_HISTORY_TURNS = 8  # keep small for latency
conversation_history = deque(maxlen=MAX_HISTORY_TURNS)  # store {"role","content"}
session_intent: Optional[str] = None

@dataclass
class EscalationPackage:
    summary: str
    intent: str
    collected_fields: dict
    best_doc_snippet: Optional[str]

def build_system_prompt(doc_text: str | None) -> str:
    """
    Compact system prompt designed for realtime usage.
    Keep it short and include the doc snippet only (not whole docs).
    """
    doc_summary = doc_text or "No policy available."
    # only insert small snippet to avoid huge prompts
    if len(doc_summary) > 1200:
        doc_summary = doc_summary[:1200] + " ...[truncated]"
    return (
        "You are Alex, a Wema Bank support assistant for TRANSFER and CARD issues only.\n"
        "Use the supplied policy snippet. Do not invent procedures beyond the policy.\n"
        f"POLICY_SNIPPET:\n{doc_summary}\n\n"
        "Output style:\n"
        "- 1 short empathetic sentence\n"
        "- concise, numbered actionable steps or minimal questions\n"
        "- if escalation required: return ESCALATE block with summary + collected fields\n"
    )

def add_to_history(role: str, content: str):
    conversation_history.append({"role": role, "content": content})

# ---------- STT utility ----------
async def speech_to_text(audio_bytes: bytes) -> str:
    """
    Non-streaming STT: push bytes into PushAudioInputStream and call recognize_once.
    Suitable for short utterances (a few seconds).
    """
    try:
        stream = speechsdk.audio.PushAudioInputStream()
        stream.write(audio_bytes)
        stream.close()
        audio_config = speechsdk.audio.AudioConfig(stream=stream)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = await recognizer.recognize_once_async()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        else:
            # Log reason for debugging
            print("STT no recognition, reason:", result.reason)
            return ""
    except Exception:
        traceback.print_exc()
        return ""

# ---------- TTS helper ----------
class TTSPushCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    def __init__(self):
        super().__init__()
        self._buffer = bytearray()

    def write(self, audio_buffer: memoryview) -> int:
        self._buffer.extend(bytes(audio_buffer))
        return len(audio_buffer)

    def close(self):
        # callback closed by SDK when done
        pass

    def get_bytes(self) -> bytes:
        return bytes(self._buffer)

def synthesize_to_bytes(text: str) -> bytes:
    """Synthesize synchronously and return raw audio bytes."""
    tts_callback = TTSPushCallback()
    tts_stream = speechsdk.audio.PushAudioOutputStream(tts_callback)
    audio_config = speechsdk.audio.AudioOutputConfig(stream=tts_stream)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return tts_callback.get_bytes()
    else:
        print("TTS synth issue, reason:", result.reason)
        return b""

# ---------- Prompting & LLM call ----------
async def generate_llm_reply_text(user_text: str, best_doc: Optional[str], client) -> Tuple[str, Optional[EscalationPackage]]:
    """
    Build a compact prompt and call your Azure OpenAI client.
    Returns (reply_text, escalation_package_or_None)
    """
    # Build system prompt
    system_prompt = build_system_prompt(best_doc)
    # Build messages (keep small)
    messages = [{"role": "system", "content": system_prompt}]
    # include last few turns
    for turn in conversation_history:
        messages.append(turn)
    messages.append({"role": "user", "content": user_text})

    # Enforce safe low-latency settings
    try:
        resp = client.chat.completions.create(
            model="aicso",
            messages=messages,
            temperature=0.0,
            max_tokens=350
        )
        reply = resp.choices[0].message.content.strip()
    except Exception:
        traceback.print_exc()
        reply = "Sorry — I couldn't generate a response. Please try again or contact support."

    # Quick parse: if assistant included an ESCALATE block, extract collected fields
    escalation = None
    if "ESCALATE" in reply or "Escalate" in reply:
        # naive parse: look for a JSON-like block or key lines
        # We'll attempt to extract collected fields from reply heuristically
        collected = {}
        # very simple heuristics: find lines like "Transaction reference: 12345"
        for line in reply.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if len(k) < 40 and len(v) < 200:
                    collected[k] = v
        escalation = EscalationPackage(
            summary=(reply[:500] + ("..." if len(reply) > 500 else "")),
            intent=session_intent or "unknown",
            collected_fields=collected,
            best_doc_snippet=(best_doc[:1000] + "..." if best_doc and len(best_doc) > 1000 else best_doc)
        )
    return reply, escalation

# ---------- Escalation persistence ----------
def persist_escalation(pkg: EscalationPackage) -> str:
    """Write escalation to file and return filepath. Adapt to DB or ticketing API."""
    os.makedirs("escalations", exist_ok=True)
    filename = f"escalations/escalation_{int(asyncio.get_event_loop().time()*1000)}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(asdict(pkg), f, ensure_ascii=False, indent=2)
    return filename

# ---------- RTC / backend hooks (replace with real implementations) ----------
async def receive_audio_callback() -> bytes:
    """Simulate capturing 5s from local mic (mono 16k int16). Replace with SignalR frame assembly."""
    duration = 5.0
    fs = 16000
    print("🎙️ [Local sim] Speak now (5s)...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    return recording.tobytes()

async def send_audio_callback(audio_bytes: bytes):
    """Simulate sending bytes to backend/SignalR. Replace with streaming send."""
    # for demo we write to file
    with open("bot_response.wav", "ab") as f:
        f.write(audio_bytes)
    print("🔊 Bot audio appended to bot_response.wav")

# ---------- Main processing loop & helper used by server ----------
async def process_audio_bytes(audio_bytes: bytes, client) -> bytes:
    """
    Full pipeline for server-side processing of audio bytes:
    - STT
    - intent classification
    - RAG retrieval
    - LLM reply generation
    - optional escalation persistence
    - TTS -> returns raw audio bytes
    """
    global session_intent

    user_text = await speech_to_text(audio_bytes)
    if not user_text:
        return synthesize_to_bytes("Sorry, I didn't catch that. Could you repeat?")

    add_to_history("user", user_text)
    intent = classify_intent(user_text)
    if session_intent != intent:
        session_intent = intent

    best_doc, score = retrieve_relevant_doc(intent, user_text)
    try:
        client_obj = client  # expected client from your config.get_azure_client()
        reply_text, escalation_pkg = await generate_llm_reply_text(user_text, best_doc, client_obj)
    except Exception:
        traceback.print_exc()
        reply_text = "Sorry — I couldn't reach the knowledge engine. Please contact support."
        escalation_pkg = None

    add_to_history("assistant", reply_text)

    if escalation_pkg:
        path = persist_escalation(escalation_pkg)
        print(f"Escalation persisted: {path}")

    # TTS
    audio_out = synthesize_to_bytes(reply_text)
    return audio_out

# ---------- Local simulator loop for manual testing ----------
async def main_loop_local_sim(client):
    # clear previous test file
    open("bot_response.wav", "wb").close()
    print("=== Voice CSO local simulator started ===")
    while True:
        try:
            audio_bytes = await receive_audio_callback()
            if not audio_bytes:
                print("No audio captured; continuing...")
                continue

            tts_bytes = await process_audio_bytes(audio_bytes, client)
            if tts_bytes:
                await send_audio_callback(tts_bytes)
                print("📁 Wrote bot_response.wav (appended)")
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting.")
            break
        except Exception:
            traceback.print_exc()
            # Continue to keep service resilient
            continue

# ---------- Example run (only for local test) ----------
if __name__ == "__main__":
    # NOTE: Replace get_azure_client() with a working client before running.
    print("This module provides process_audio_bytes(...) to integrate into your RTC/SignalR flow.")
    # Example local sim (will raise NotImplementedError for get_azure_client)
    try:
        client = get_azure_client()
        asyncio.run(main_loop_local_sim(client))
    except NotImplementedError:
        print("Please plug in your Azure OpenAI client in get_azure_client() and re-run.")
