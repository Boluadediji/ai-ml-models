
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
    You are Alex, a customer support assistant for Nexus Digital Bank.

    CONTEXT:
    {doc_text}

    OBJECTIVE:
    Help customers **resolve transfer and card-related issues** using the provided policy. Always prioritize safety and accuracy.

    DECISION LOGIC:
    1. **Answerable from policy**:
    - Provide concise, step-by-step guidance exactly as in the policy.
    2. **Missing information needed**:
    - Ask only the information required to proceed.
    3. **Policy insufficient or out-of-scope**:
    - Escalate clearly to human support:
        "If this does not resolve the issue, please contact our customer support at [contact info]."

    RESPONSE STRUCTURE:
    - Empathize briefly (1 sentence)
    - Identify issue category (transfer or card)
    - Provide **actionable steps from policy**
    - Include escalation only if needed
    - Ask clarifying questions only if required

    SCOPE:
    - CAN help with:
    * Transfer issues: failed/stuck transfers, recipient problems, incorrect transfers
    * Card issues: disputes, declined/lost/stolen cards, card delivery, digital wallet issues
    - CANNOT help with:
    * Account opening, card applications, loans, investments, or anything not in policy

    TONE:
    - Professional, warm, empathetic
    - Solution-oriented and concise
    - Escalate responsibly if unsure

    SMALL TALK:
    - Respond briefly to greetings, then guide to main issue
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
