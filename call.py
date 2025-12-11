import os
import json
import asyncio
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

from intent_classifier import classify_intent
from ragengine import retrieve_relevant_doc
from config import get_azure_client

# --------------------------
# ENV / Azure Speech
# --------------------------
load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")
if not (AZURE_SPEECH_KEY and AZURE_REGION):
    raise RuntimeError("Set AZURE_SPEECH_KEY / AZURE_SPEECH_REGION")

speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
speech_config.speech_recognition_language = "en-NG"
speech_config.speech_synthesis_voice_name = "en-NG-AdaNeural"

# --------------------------
# Session state
# --------------------------
conversation_history = deque(maxlen=8)
session_intent: Optional[str] = None

@dataclass
class EscalationPackage:
    summary: str
    intent: str
    collected_fields: dict
    best_doc_snippet: Optional[str]

# --------------------------
# Streaming TTS class
# --------------------------
class TTSPushCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    def __init__(self, send_chunk_callback):
        super().__init__()
        self.send_chunk = send_chunk_callback  # function to send audio chunk
    def write(self, audio_buffer: memoryview) -> int:
        self.send_chunk(bytes(audio_buffer))
        return len(audio_buffer)
    def close(self):
        pass

# --------------------------
# Live Call Handler
# --------------------------
async def handle_live_call(get_audio_chunk, send_audio_chunk, client):
    """
    get_audio_chunk(): async generator yielding raw audio bytes from user
    send_audio_chunk(bytes): send TTS audio chunk to the user
    """
    global session_intent
    buffer = bytearray()
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=None)
    stop_recognition = asyncio.Event()

    # Event handler for partial recognition
    def recognized(evt):
        nonlocal buffer
        text = evt.result.text.strip()
        if text:
            asyncio.create_task(process_user_utterance(text, client, send_audio_chunk))

    recognizer.recognized.connect(recognized)
    
    # Start continuous recognition
    recognizer.start_continuous_recognition()

    async for chunk in get_audio_chunk():
        # Send audio to recognizer
        # Here you could implement PushAudioInputStream for live feeding
        buffer.extend(chunk)

    await stop_recognition.wait()
    recognizer.stop_continuous_recognition()

async def process_user_utterance(user_text: str, client, send_audio_chunk):
    global session_intent
    conversation_history.append({"role": "user", "content": user_text})

    # Intent
    intent = classify_intent(user_text)
    if session_intent != intent:
        session_intent = intent

    # RAG
    best_doc, _ = retrieve_relevant_doc(intent, user_text)

    # LLM prompt
    system_prompt = f"""
You are Alex, a Wema Bank support assistant (TRANSFER and CARD issues only).
Use the supplied policy document and do not invent procedures.

POLICY_DOCUMENT:
{best_doc}

Output style:
- 1 short empathetic sentence max
- concise actionable steps or minimal safe questions
- append <<ESCALATE>> only if escalation required
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="aicso",
        messages=messages,
        temperature=0.0,
        max_tokens=350
    )
    raw_reply = resp.choices[0].message.content.strip()

    # Escalation
    if "<<ESCALATE>>" in raw_reply:
        collected = {}
        for line in raw_reply.splitlines():
            if ":" in line:
                collected[line.split(":",1)[0].strip()] = line.split(":",1)[1].strip()
        pkg = EscalationPackage(
            summary=raw_reply[:500],
            intent=session_intent,
            collected_fields=collected,
            best_doc_snippet=best_doc
        )
        os.makedirs("escalations", exist_ok=True)
        filename = f"escalations/escalation_{int(asyncio.get_event_loop().time()*1000)}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(asdict(pkg), f, ensure_ascii=False, indent=2)
        raw_reply = raw_reply.replace("<<ESCALATE>>","").strip()

    conversation_history.append({"role": "assistant", "content": raw_reply})

    # Streaming TTS
    tts_callback = TTSPushCallback(send_audio_chunk)
    tts_stream = speechsdk.audio.PushAudioOutputStream(tts_callback)
    audio_config = speechsdk.audio.AudioOutputConfig(stream=tts_stream)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(raw_reply).get()
