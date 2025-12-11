"""
Realtime voice RAG agent for transfer & card issues.

- Plug `process_audio_bytes` into your backend/RTC layer.
- Requires Azure OpenAI client via get_azure_client().
- Set AZURE_SPEECH_KEY / AZURE_SPEECH_REGION in your env.
"""

import asyncio
import os
import json
import traceback
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

from intent_classifier import classify_intent
from ragengine import retrieve_relevant_doc
from config import get_azure_client

# --------------------------
# Env / Azure Speech
# --------------------------
load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")

if not (AZURE_SPEECH_KEY and AZURE_REGION):
    raise RuntimeError("Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in your env")

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
# STT
# --------------------------
async def speech_to_text(audio_bytes: bytes) -> str:
    try:
        stream = speechsdk.audio.PushAudioInputStream()
        stream.write(audio_bytes)
        stream.close()
        audio_config = speechsdk.audio.AudioConfig(stream=stream)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = await recognizer.recognize_once_async()
        return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else ""
    except Exception:
        traceback.print_exc()
        return ""

# --------------------------
# TTS
# --------------------------
class TTSPushCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    def __init__(self):
        super().__init__()
        self._buffer = bytearray()
    def write(self, audio_buffer: memoryview) -> int:
        self._buffer.extend(bytes(audio_buffer))
        return len(audio_buffer)
    def close(self):
        pass
    def get_bytes(self) -> bytes:
        return bytes(self._buffer)

def synthesize_to_bytes(text: str) -> bytes:
    callback = TTSPushCallback()
    tts_stream = speechsdk.audio.PushAudioOutputStream(callback)
    audio_config = speechsdk.audio.AudioOutputConfig(stream=tts_stream)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    return callback.get_bytes() if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else b""

# --------------------------
# Process audio
# --------------------------
async def process_audio_bytes(audio_bytes: bytes, client) -> bytes:
    global session_intent

    # STT
    user_text = await speech_to_text(audio_bytes)
    if not user_text:
        return synthesize_to_bytes("Sorry, I didn't catch that. Could you repeat?")

    conversation_history.append({"role": "user", "content": user_text})

    # Intent
    intent = classify_intent(user_text)
    if session_intent != intent:
        session_intent = intent

    # RAG
    best_doc, _ = retrieve_relevant_doc(intent, user_text)

    # LLM prompt
    system_prompt = f"""
    You are Alex, a customer support assistant for Wema Bank (ALAT).  
    You provide chat and voice support for TRANSFER and CARD issues only.

    BEHAVIOR:
    - Always attempt to **resolve the user’s issue first** using policy and RAG context.  
    - Ask **only the minimum required details**.  
    - Keep responses short, human-like, and empathetic.  
    - Do not guess or invent procedures outside policy.  
    - Never ask for sensitive info (BVN, PIN, OTP, NIN, card numbers, passwords).

    CONTEXT:
    {best_doc}

    -------------------------
    1. GUIDED WORKFLOW
    -------------------------
    1. Check the policy context (RAG document) to see if the issue can be resolved.  
    2. If yes:
    - Provide **step-by-step guidance**.  
    - Ask for any missing but safe details (transaction reference, amount, date).  
    3. If info is missing:
    - Ask only for safe, required info.  
    4. Only escalate when:
    - Issue cannot be solved with the policy.  
    - Sensitive info is shared or requested.  
    - Repeated failures or urgent/fraud cases occur.

    -------------------------
    2. SENSITIVE-DATA RULES
    -------------------------
    - Stop and escalate if the user mentions or shares: BVN, PIN, OTP, passwords, full card numbers, NIN.  
    - Include a gentle warning:  
    "For your safety, please don’t share BVN, PIN, OTP, or card details here."  

    Escalation response should be:  
    "Let me run a quick check… Based on what you shared, this needs a secure review. I’m connecting you to a support specialist."  
    Append <<ESCALATE>> at the end.

    -------------------------
    3. OTHER ESCALATION CASES
    -------------------------
    - Fraud, scam, unauthorized transaction  
    - Debit but no cash  
    - Lost/stolen card  
    - Repeated failed transfers  
    - Account compromise  
    - Distressed user or urgent issue

    Include agent-like explanation and <<ESCALATE>>.

    -------------------------
    4. RESPONSE STYLE
    -------------------------
    - 1 short empathetic sentence max.  
    - Clear actions or questions.  
    - Avoid generic “contact customer service”.  
    - Use dynamic escalation language:  
    - "Let me run a quick check…"  
    - "Based on your info, I’m connecting you to a specialist."  
    - "They’ll reach out shortly with an update."

    -------------------------
    5. OUTPUT
    -------------------------
    - Return only the **natural-language reply**.  
    - Append <<ESCALATE>> **only if escalation is triggered**.  
    - If resolving, give step-by-step guidance or safe follow-ups.

    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    try:
        resp = client.chat.completions.create(
            model="aicso",
            messages=messages,
            temperature=0.0,
            max_tokens=350
        )
        raw_reply = resp.choices[0].message.content.strip()
    except Exception:
        traceback.print_exc()
        raw_reply = "Sorry — I couldn't generate a response. Please try again."

    # Escalation
    escalation_pkg: Optional[EscalationPackage] = None
    if "<<ESCALATE>>" in raw_reply:
        collected = {}
        for line in raw_reply.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                collected[k.strip()] = v.strip()
        escalation_pkg = EscalationPackage(
            summary=raw_reply[:500],
            intent=session_intent or "unknown",
            collected_fields=collected,
            best_doc_snippet=best_doc
        )
        raw_reply = raw_reply.replace("<<ESCALATE>>", "").strip()

        # Persist escalation
        os.makedirs("escalations", exist_ok=True)
        filename = f"escalations/escalation_{int(asyncio.get_event_loop().time()*1000)}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(asdict(escalation_pkg), f, ensure_ascii=False, indent=2)
        print(f"Escalation saved: {filename}")

    conversation_history.append({"role": "assistant", "content": raw_reply})

    # TTS
    return synthesize_to_bytes(raw_reply)
