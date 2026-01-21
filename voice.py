import asyncio
import traceback
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

from config import get_azure_client
from text_handler import handle_text_message
from intent_classifier import classify_intent
from ragengine import retrieve_relevant_doc

load_dotenv()

# ------------------------------
# ENV / Azure Speech
# ------------------------------
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")

if not (AZURE_SPEECH_KEY and AZURE_REGION):
    raise RuntimeError("Missing Azure STT/TTS credentials")

speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY,
    region=AZURE_REGION
)
speech_config.speech_recognition_language = "en-NG"
speech_config.speech_synthesis_voice_name = "en-NG-AdaNeural"

client = get_azure_client()


# ===========================================================
#  SPEECH → TEXT
# ===========================================================
async def speech_to_text(audio_bytes: bytes) -> str:
    """
    Converts raw PCM/WAV audio bytes to text using Azure STT.
    """
    try:
        stream = speechsdk.audio.PushAudioInputStream()
        stream.write(audio_bytes)
        stream.close()

        audio = speechsdk.audio.AudioConfig(stream=stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio
        )

        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text or ""
        return ""
    except Exception:
        traceback.print_exc()
        return ""


# ===========================================================
#  TEXT → SPEECH (TTS)
# ===========================================================
async def text_to_speech(text: str) -> bytes:
    """
    Synthesizes text to speech and returns raw wav bytes.
    """
    try:
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data

        return b""
    except Exception:
        traceback.print_exc()
        return b""


# ===========================================================
#  PROCESS ONE VOICE TURN (API)
# ===========================================================
async def process_user_utterance(audio_bytes: bytes) -> dict:
    """
    Main voice pipeline for API:
    1. STT
    2. Intent
    3. RAG
    4. handle_text_message()
    5. TTS
    """

    # 1. ASR
    transcript = await speech_to_text(audio_bytes)

    if not transcript:
        return {
            "text": "",
            "intent": "no_speech",
            "reply": "Sorry, I didn't catch that. Please try again.",
            "audio": await text_to_speech("Sorry, I didn't catch that. Please try again.")
        }

    # 2. Classification
    try:
        intent = classify_intent(transcript)
    except Exception:
        intent = "unsupported_request"

    # 3. RAG
    try:
        best_doc, _ = retrieve_relevant_doc(intent, transcript)
    except Exception:
        best_doc = None

    # 4. Same logic as text chat
    temp_session = {
        "history": [],
        "intent": intent,
        "session_intent": intent
    }

    reply, escalation, updated_session = await handle_text_message(
        transcript,
        temp_session,
        injected_rag_doc=best_doc
    )

    # 5. TTS
    audio = await text_to_speech(reply)

    return {
        "text": transcript,
        "intent": updated_session.get("intent", intent),
        "reply": reply,
        "audio": audio
    }


# ===========================================================
#  STREAMING LIVE CALL HANDLER (WebSocket)
# ===========================================================
async def handle_live_call(get_audio_chunk, send_audio_chunk, client):
    """
    True streaming phone-call behavior.

    get_audio_chunk(): async generator → yields mic audio bytes
    send_audio_chunk(bytes): async → sends TTS bytes back to caller
    """

    print("📞 Live call started...")

    async for chunk in get_audio_chunk():
        try:
            # Convert speech to text
            transcript = await speech_to_text(chunk)
            if not transcript:
                continue

            print("👤 USER SAID:", transcript)

            # Intent
            try:
                intent = classify_intent(transcript)
            except:
                intent = "unsupported_request"

            # RAG
            try:
                doc, _ = retrieve_relevant_doc(intent, transcript)
            except:
                doc = None

            # Process with full text-handler logic
            temp_session = {"history": [], "intent": intent}
            reply, escalation, updated = await handle_text_message(
                transcript,
                temp_session,
                injected_rag_doc=doc
            )

            print("🤖 ALEX:", reply)

            # TTS streaming output
            tts_audio = await text_to_speech(reply)
            await send_audio_chunk(tts_audio)

        except Exception:
            traceback.print_exc()
            continue

    print("📵 Live call ended.")


if __name__ == "__main__":
    import asyncio

    async def main():
        # Load a test WAV file
        test_wav_path = "test_audio.wav"  # replace with your file
        if not os.path.exists(test_wav_path):
            print(f"Test audio file not found: {test_wav_path}")
            return

        with open(test_wav_path, "rb") as f:
            audio_bytes = f.read()

        # Process the user utterance
        result = await process_user_utterance(audio_bytes)

        print("Transcript:", result["text"])
        print("Intent:", result["intent"])
        print("Reply:", result["reply"])
        print("Audio bytes length:", len(result["audio"]))

        # Optional: save TTS output to a file
        with open("tts_output.wav", "wb") as f:
            f.write(result["audio"])
        print("TTS audio saved as tts_output.wav")

    asyncio.run(main())