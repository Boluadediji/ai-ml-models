import os
import traceback
import wave
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import pyaudio
import numpy as np

from text_handler import handle_text_message
from intent_classifier import classify_intent
from ragengine import retrieve_relevant_doc

load_dotenv()

# Debug: Check environment variables
print("🔍 Checking environment variables...")
print(f"AZURE_SPEECH_KEY exists: {bool(os.getenv('AZURE_SPEECH_KEY'))}")
print(f"AZURE_SPEECH_REGION: {os.getenv('AZURE_SPEECH_REGION')}")

# =========================================================
# Azure Speech Config
# =========================================================
try:
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION")
    )
    speech_config.speech_recognition_language = "en-NG"
    speech_config.speech_synthesis_voice_name = "en-NG-AdaNeural"
    print("✅ Speech config initialized")
except Exception as e:
    print(f"❌ Failed to initialize speech config: {e}")
    exit(1)

# =========================================================
# SPEECH → TEXT - FIXED VERSION
# =========================================================

def speech_to_text_from_wav(audio_bytes: bytes) -> str:
    """Process WAV audio bytes specifically"""
    try:
        # First, let's save the bytes to a temporary file and read it properly
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
        
        # Read the WAV file properly
        with wave.open(temp_file, 'rb') as wav_file:
            print(f"WAV file info: {wav_file.getnchannels()} channels, "
                  f"{wav_file.getsampwidth()} bytes/sample, "
                  f"{wav_file.getframerate()} Hz")
            
            # Check if format is supported
            if wav_file.getnchannels() > 2:
                print("⚠️ Warning: More than 2 channels, converting to mono")
            
            if wav_file.getsampwidth() != 2:
                print("⚠️ Warning: Sample width is not 16-bit (2 bytes)")
        
        # Use AudioConfig from file
        audio_config = speechsdk.audio.AudioConfig(filename=temp_file)
        
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        print("🎤 Starting speech recognition...")
        result = recognizer.recognize_once_async().get()
        
        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = result.text.strip()
            print(f"✅ Recognized: {text}")
            return text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            details = speechsdk.NoMatchDetails.from_result(result)
            print(f"❌ No speech recognized: {details}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = speechsdk.CancellationDetails.from_result(result)
            print(f"❌ Recognition canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation.error_details}")
                
    except Exception as e:
        print(f"❌ Exception in speech recognition: {e}")
        traceback.print_exc()
    
    return ""

def speech_to_text(audio_bytes: bytes) -> str:
    """Universal speech recognition"""
    try:
        print("🎤 Trying audio data stream approach...")
        
        # Create a push stream
        format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000, 
            bits_per_sample=16, 
            channels=1
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Start recognition in background
        recognition_done = False
        recognized_text = ""
        
        def recognized_callback(evt):
            nonlocal recognized_text, recognition_done
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                recognized_text = evt.result.text
                print(f"✅ Recognized in callback: {recognized_text}")
            recognition_done = True
        
        def canceled_callback(evt):
            nonlocal recognition_done
            print(f"Recognition canceled: {evt}")
            recognition_done = True
        
        # Connect events
        recognizer.recognized.connect(recognized_callback)
        recognizer.canceled.connect(canceled_callback)
        
        # Start continuous recognition
        recognizer.start_continuous_recognition_async().get()
        
        # Push audio data
        print(f"📤 Pushing {len(audio_bytes)} bytes of audio...")
        push_stream.write(audio_bytes)
        push_stream.close()
        
        # Wait for recognition
        import time
        timeout = 10  # seconds
        start_time = time.time()
        
        while not recognition_done and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        recognizer.stop_continuous_recognition_async().get()
        
        if recognition_done:
            return recognized_text
        else:
            print("⏱️ Recognition timeout")
            
    except Exception as e:
        print(f"❌ Error in speech_to_text: {e}")
        traceback.print_exc()
    
    # Fallback: Try WAV-specific method
    print("🔄 Falling back to WAV-specific method...")
    return speech_to_text_from_wav(audio_bytes)

# =========================================================
# TEXT → SPEECH
# =========================================================

def text_to_speech(text: str) -> bytes:
    try:
        # Use async method
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )
        
        print(f"🔊 Synthesizing: {text[:50]}...")
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            print(f"✅ Synthesized {len(audio_data)} bytes")
            return audio_data
        else:
            print(f"❌ Synthesis failed: {result.reason}")
            
    except Exception as e:
        print(f"❌ Exception in TTS: {e}")
        traceback.print_exc()
    
    return b""

# =========================================================
# MAIN VOICE CSO PIPELINE
# =========================================================

async def voice_cso(audio_bytes: bytes) -> dict:
    """
    AI-only voice CSO pipeline:
    STT → Intent → RAG → LLM → TTS
    """
    
    print("\n" + "="*50)
    print("STARTING VOICE CSO PIPELINE")
    print("="*50)
    
    # 1. Speech → Text
    print("\n🔍 Step 1: Speech to Text...")
    transcript = speech_to_text(audio_bytes)
    
    if not transcript:
        reply = "Sorry, I didn't catch that. Could you please speak again?"
        print(f"⚠️ No transcript. Default reply: {reply}")
        return {
            "transcript": "",
            "intent": "no_speech",
            "reply": reply,
            "audio": text_to_speech(reply)
        }
    
    # 2. Intent Classification
    print("\n🔍 Step 2: Intent Classification...")
    try:
        intent = classify_intent(transcript)
        print(f"✅ Intent: {intent}")
    except Exception as e:
        print(f"❌ Intent classification error: {e}")
        intent = "unsupported_request"
    
    # 3. RAG
    print("\n🔍 Step 3: RAG Retrieval...")
    try:
        rag_doc, _ = retrieve_relevant_doc(intent, transcript)
        if rag_doc:
            print(f"✅ RAG document retrieved ({len(rag_doc)} chars)")
        else:
            print("ℹ️ No RAG document found")
    except Exception as e:
        print(f"❌ RAG error: {e}")
        rag_doc = None
    
    # 4. LLM
    print("\n🔍 Step 4: LLM Processing...")
    session = {"history": [], "intent": intent}
    
    try:
        reply, escalation, _ = await handle_text_message(
            transcript,
            session,
            injected_rag_doc=rag_doc
        )
        print(f"✅ LLM reply generated: {reply[:100]}...")
    except Exception as e:
        print(f"❌ LLM error: {e}")
        reply = "I'm having trouble processing your request right now. Please try again."
        escalation = None
    
    # 5. Text → Speech
    print("\n🔍 Step 5: Text to Speech...")
    audio = text_to_speech(reply)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    
    return {
        "transcript": transcript,
        "intent": intent,
        "reply": reply,
        "audio": audio
    }

# =========================================================
# TEST FUNCTION
# =========================================================

def create_test_audio():
    """Create a simple test WAV file"""
    print("Creating test audio file...")
    
    # Create 2 seconds of audio saying "Hello"
    sample_rate = 16000
    duration = 2.0
    
    # Create a simple tone
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a simple "Hello" pattern (very basic)
    tone1 = np.sin(2 * np.pi * 300 * t[:len(t)//4]) * 0.3  # "He"
    tone2 = np.sin(2 * np.pi * 250 * t[len(t)//4:len(t)//2]) * 0.3  # "llo"
    tone3 = np.sin(2 * np.pi * 200 * t[len(t)//2:]) * 0.3  # silence
    
    audio_data = np.concatenate([tone1, tone2, tone3])
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV
    with wave.open("test_hello.wav", 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✅ Created test_hello.wav ({len(audio_data)} samples)")
    return "test_hello.wav"

if __name__ == "__main__":
    import asyncio
    
    async def main():
        TEST_AUDIO = "test_audio.wav"
        
        # Check if test file exists
        if not os.path.exists(TEST_AUDIO):
            print(f"⚠️ {TEST_AUDIO} not found.")
            print("Choose an option:")
            print("1. Use simple test audio (says 'Hello')")
            print("2. Enter path to your audio file")
            
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                TEST_AUDIO = create_test_audio()
            elif choice == "2":
                TEST_AUDIO = input("Enter audio file path: ").strip()
                if not os.path.exists(TEST_AUDIO):
                    print(f"❌ File not found: {TEST_AUDIO}")
                    TEST_AUDIO = create_test_audio()
            else:
                TEST_AUDIO = create_test_audio()
        
        # Load audio file
        print(f"\n📁 Loading audio file: {TEST_AUDIO}")
        try:
            with open(TEST_AUDIO, "rb") as f:
                audio_bytes = f.read()
            print(f"✅ Loaded {len(audio_bytes)} bytes")
            
            # Process
            result = await voice_cso(audio_bytes)
            
            # Display results
            print("\n" + "="*50)
            print("FINAL RESULTS")
            print("="*50)
            print(f"📝 Transcript: {result['transcript']}")
            print(f"🎯 Intent: {result['intent']}")
            print(f"💬 Reply: {result['reply']}")
            print(f"🔊 Audio size: {len(result['audio'])} bytes")
            
            # Save output
            if result['audio']:
                output_file = "voice_reply.wav"
                with open(output_file, "wb") as f:
                    f.write(result['audio'])
                print(f"\n💾 Reply saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Main error: {e}")
            traceback.print_exc()
    
    # Run with timeout
    try:
        asyncio.run(main(), debug=True)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()