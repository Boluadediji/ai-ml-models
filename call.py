# import os
# import json
# import asyncio
# from collections import deque
# from dataclasses import dataclass, asdict
# from typing import Optional

# import azure.cognitiveservices.speech as speechsdk
# from dotenv import load_dotenv

# from intent_classifier import classify_intent
# from ragengine import retrieve_relevant_doc
# from config import get_azure_client

# # --------------------------
# # ENV / Azure Speech
# # --------------------------
# load_dotenv()
# AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
# AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")
# if not (AZURE_SPEECH_KEY and AZURE_REGION):
#     raise RuntimeError("Set AZURE_SPEECH_KEY / AZURE_SPEECH_REGION")

# speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
# speech_config.speech_recognition_language = "en-NG"
# speech_config.speech_synthesis_voice_name = "en-NG-AdaNeural"

# # --------------------------
# # Session state
# # --------------------------
# conversation_history = deque(maxlen=8)
# session_intent: Optional[str] = None

# @dataclass
# class EscalationPackage:
#     summary: str
#     intent: str
#     collected_fields: dict
#     best_doc_snippet: Optional[str]

# # --------------------------
# # Streaming TTS class
# # --------------------------
# class TTSPushCallback(speechsdk.audio.PushAudioOutputStreamCallback):
#     def __init__(self, send_chunk_callback):
#         super().__init__()
#         self.send_chunk = send_chunk_callback  # function to send audio chunk
#     def write(self, audio_buffer: memoryview) -> int:
#         self.send_chunk(bytes(audio_buffer))
#         return len(audio_buffer)
#     def close(self):
#         pass

# # --------------------------
# # Live Call Handler
# # --------------------------
# async def handle_live_call(get_audio_chunk, send_audio_chunk, client):
#     """
#     get_audio_chunk(): async generator yielding raw audio bytes from user
#     send_audio_chunk(bytes): send TTS audio chunk to the user
#     """
#     global session_intent
#     buffer = bytearray()
#     recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=None)
#     stop_recognition = asyncio.Event()

#     # Event handler for partial recognition
#     def recognized(evt):
#         nonlocal buffer
#         text = evt.result.text.strip()
#         if text:
#             asyncio.create_task(process_user_utterance(text, client, send_audio_chunk))

#     recognizer.recognized.connect(recognized)
    
#     # Start continuous recognition
#     recognizer.start_continuous_recognition()

#     async for chunk in get_audio_chunk():
#         # Send audio to recognizer
#         # Here you could implement PushAudioInputStream for live feeding
#         buffer.extend(chunk)

#     await stop_recognition.wait()
#     recognizer.stop_continuous_recognition()

# async def process_user_utterance(user_text: str, client, send_audio_chunk):
#     global session_intent
#     conversation_history.append({"role": "user", "content": user_text})

#     # Intent
#     intent = classify_intent(user_text)
#     if session_intent != intent:
#         session_intent = intent

#     # RAG
#     best_doc, _ = retrieve_relevant_doc(intent, user_text)

#     # LLM prompt
#     system_prompt = f"""
# You are Alex, a Wema Bank support assistant (TRANSFER and CARD issues only).
# Use the supplied policy document and do not invent procedures.

# POLICY_DOCUMENT:
# {best_doc}

# Output style:
# - 1 short empathetic sentence max
# - concise actionable steps or minimal safe questions
# - append <<ESCALATE>> only if escalation required
# """
#     messages = [{"role": "system", "content": system_prompt}]
#     messages.extend(conversation_history)
#     messages.append({"role": "user", "content": user_text})

#     resp = client.chat.completions.create(
#         model="aicso",
#         messages=messages,
#         temperature=0.0,
#         max_tokens=350
#     )
#     raw_reply = resp.choices[0].message.content.strip()

#     # Escalation
#     if "<<ESCALATE>>" in raw_reply:
#         collected = {}
#         for line in raw_reply.splitlines():
#             if ":" in line:
#                 collected[line.split(":",1)[0].strip()] = line.split(":",1)[1].strip()
#         pkg = EscalationPackage(
#             summary=raw_reply[:500],
#             intent=session_intent,
#             collected_fields=collected,
#             best_doc_snippet=best_doc
#         )
#         os.makedirs("escalations", exist_ok=True)
#         filename = f"escalations/escalation_{int(asyncio.get_event_loop().time()*1000)}.json"
#         with open(filename, "w", encoding="utf-8") as f:
#             json.dump(asdict(pkg), f, ensure_ascii=False, indent=2)
#         raw_reply = raw_reply.replace("<<ESCALATE>>","").strip()

#     conversation_history.append({"role": "assistant", "content": raw_reply})

#     # Streaming TTS
#     tts_callback = TTSPushCallback(send_audio_chunk)
#     tts_stream = speechsdk.audio.PushAudioOutputStream(tts_callback)
#     audio_config = speechsdk.audio.AudioOutputConfig(stream=tts_stream)
#     synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
#     synthesizer.speak_text_async(raw_reply).get()



import asyncio
import traceback
import sounddevice as sd
import azure.cognitiveservices.speech as speechsdk
from intent_classifier import classify_intent
from ragengine import retrieve_relevant_doc
from config import get_azure_client
from dotenv import load_dotenv
import os

load_dotenv()

# ------------------------------
# ENV / Azure clients
# ------------------------------
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")
if not (AZURE_SPEECH_KEY and AZURE_REGION):
    raise RuntimeError("Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in your env")

client = get_azure_client()  # your azure-openai wrapper client

# ------------------------------
# Mock / RTC audio callbacks (replace with your SignalR/RTC hooks)
# ------------------------------
async def receive_audio_callback():
    """Simulate incoming RTC audio (mic capture)"""
    duration = 5
    fs = 16000
    print("\n🎙️ Speak now (capturing 5s)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    return audio.tobytes()

async def send_audio_callback(audio_bytes):
    """Simulate sending outgoing audio to backend (append to a file)"""
    # Replace this with sending bytes to SignalR / RTC channel
    with open("bot_response.wav", "ab") as f:
        f.write(audio_bytes)
    print("🔊 Wrote TTS bytes to bot_response.wav")

# ------------------------------
# Azure Speech SDK config
# ------------------------------
speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
speech_config.speech_recognition_language = "en-NG"
speech_config.speech_synthesis_voice_name = "en-NG-AdaNeural"

# ------------------------------
# TTS stream callback (synchronous write)
# ------------------------------
class TTSPushCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    """
    Simple PushAudioOutputStreamCallback that writes each chunk to a file.
    This avoids asyncio/thread loop complexity; replace with your SignalR sender if needed.
    """
    def __init__(self, out_filename="bot_response.wav"):
        super().__init__()
        self.out_filename = out_filename
        # clear file on init
        open(self.out_filename, "wb").close()

    def write(self, audio_buffer: memoryview) -> int:
        # audio_buffer is a memoryview; write raw bytes to file
        with open(self.out_filename, "ab") as f:
            f.write(bytes(audio_buffer))
        return len(audio_buffer)

    def close(self):
        print("✅ TTS PushAudioOutputStream closed")

tts_callback = TTSPushCallback(out_filename="bot_response.wav")
tts_stream = speechsdk.audio.PushAudioOutputStream(tts_callback)
tts_audio_config = speechsdk.audio.AudioOutputConfig(stream=tts_stream)

synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=tts_audio_config)

# ------------------------------
# Globals for conversation state
# ------------------------------
conversation_history = []     # store user + assistant turns (minimal)
session_intent = None         # store current session intent

# ------------------------------
# LLM + RAG response (returns reply text)
# ------------------------------
async def generate_llm_reply(user_text: str, best_doc: str | None) -> str:
    """
    Builds system prompt using best_doc (if present) and calls your Azure OpenAI client.
    Returns assistant reply text.
    """
    global conversation_history

    # Build prompt using the retrieved doc (if any)
    doc_text = best_doc if best_doc else "No policy doc available."

    SYSTEM_PROMPT = f"""
    You are Alex, a customer support assistant for Wema Bank (ALAT).  
    You provide chat and voice support for TRANSFER and CARD issues only.

    BEHAVIOR:
    - Always attempt to **resolve the user’s issue first** using policy and RAG context.  
    - Ask **only the minimum required details**.  
    - Keep responses short, human-like, and empathetic.  
    - Do not guess or invent procedures outside policy.  
    - Never ask for sensitive info (BVN, PIN, OTP, NIN, card numbers, passwords).

    CONTEXT:
    {doc_text}

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
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="aicso",
            messages=messages,
            temperature=0,
            max_tokens=300
        )
        reply = response.choices[0].message.content.strip()
        # Save to conversation history
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception:
        print("Error calling LLM:")
        traceback.print_exc()
        return "Sorry — I couldn't produce an answer at the moment. Please try again or contact support."

# ------------------------------
# Speech-to-text helper (non-streaming)
# ------------------------------
async def speech_to_text(audio_bytes: bytes) -> str:
    stream = speechsdk.audio.PushAudioInputStream()
    stream.write(audio_bytes)
    stream.close()

    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        # For debugging print reason
        # print("STT reason:", result.reason)
        return ""

# ------------------------------
# Main loop
# ------------------------------
async def main():
    global session_intent

    # ensure previous bot audio cleared
    open("bot_response.wav", "wb").close()

    print("=== Voice CSO local simulator started ===")
    while True:
        try:
            audio = await receive_audio_callback()
            if not audio:
                print("No audio captured; continuing...")
                continue

            user_text = await speech_to_text(audio)
            if not user_text:
                print("❌ No speech recognized.")
                continue

            print(f"👤 USER: {user_text}")

            # Intent classification
            try:
                intent = classify_intent(user_text)
                print(f"(Detected intent: {intent})")
                if session_intent != intent:
                    print(f"(Switching session intent from {session_intent} → {intent})")
                    session_intent = intent
            except Exception:
                print("⚠️ Intent classification failed; marking as unsupported_request")
                session_intent = "unsupported_request"

            # RAG retrieval
            try:
                best_doc, score = retrieve_relevant_doc(session_intent, user_text)
                print(f"(Retrieved doc score: {score})")
            except Exception:
                print("⚠️ RAG retrieval failed.")
                traceback.print_exc()
                best_doc = None

            # Generate LLM reply
            bot_text = await generate_llm_reply(user_text, best_doc)
            print(f"🤖 BOT: {bot_text}")

            # Synthesize — this writes audio bytes chunk-by-chunk to bot_response.wav via our callback
            synth_result = synthesizer.speak_text_async(bot_text).get()
            # Optionally you can inspect synth_result.reason, etc.
            if synth_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("✅ Synthesized audio written to bot_response.wav")
            else:
                print("⚠️ TTS finished with reason:", synth_result.reason)

            # If you want to immediately "send" the produced audio to your backend,
            # either read bot_response.wav or have your callback directly stream to SignalR.
            # For demo, we will simply print the file path:
            print("📁 TTS file: bot_response.wav")

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting.")
            break
        except Exception:
            print("Unexpected error in main loop:")
            traceback.print_exc()
            # decide whether to continue or break — continue for resilience
            continue


async def process_audio_bytes(audio_bytes: bytes) -> bytes:
    """
    1. STT
    2. Intent + RAG
    3. LLM
    4. TTS
    5. return raw audio bytes
    """
    user_text = await speech_to_text(audio_bytes)
    intent = classify_intent(user_text)
    doc, _ = retrieve_relevant_doc(intent, user_text)
    reply = await generate_llm_reply(user_text, doc)

    tts_result = synthesizer.speak_text_async(reply).get()
    return tts_result.audio_data



if __name__ == "__main__":
    asyncio.run(main())
