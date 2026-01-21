import pyaudio
import wave
import threading
import time
import numpy as np
from datetime import datetime
import os

class MicrophoneRecorder:
    def __init__(self, format=pyaudio.paInt16, channels=1, rate=16000, chunk=1024):
        """
        Initialize microphone recorder
        
        Args:
            format: Audio format (paInt16 for 16-bit PCM)
            channels: Number of audio channels
            rate: Sample rate (Hz)
            chunk: Chunk size for recording
        """
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.frames = []
        self.stream = None
        
    def start_recording(self, duration=None):
        """
        Start recording from microphone
        
        Args:
            duration: Recording duration in seconds (None for continuous)
        """
        self.is_recording = True
        self.frames = []
        
        # Open stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("🎤 Recording started... (Press Enter to stop)")
        
        # Record for specified duration or until stopped
        if duration:
            start_time = time.time()
            while time.time() - start_time < duration:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
        else:
            # Record until stop_recording is called
            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Recording error: {e}")
                    break
        
        # Cleanup
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        return self.get_audio_bytes()
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        
    def get_audio_bytes(self):
        """Get recorded audio as bytes"""
        if not self.frames:
            return b""
        
        # Convert frames to single bytes object
        audio_bytes = b''.join(self.frames)
        return audio_bytes
    
    def save_to_wav(self, filename=None):
        """Save recorded audio to WAV file"""
        if not self.frames:
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
        
        # Ensure directory exists
        os.makedirs("recordings", exist_ok=True)
        filepath = os.path.join("recordings", filename)
        
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        print(f"💾 Audio saved to: {filepath}")
        return filepath
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.audio:
            self.audio.terminate()

def record_from_microphone(duration=None):
    """
    Simple function to record from microphone
    
    Args:
        duration: Recording duration in seconds
        
    Returns:
        bytes: Audio data
    """
    recorder = MicrophoneRecorder()
    
    try:
        print(f"\n{'='*50}")
        print("VOICE CSO - MICROPHONE TEST")
        print("="*50)
        print(f"Recording settings:")
        print(f"  Sample rate: {recorder.rate} Hz")
        print(f"  Channels: {recorder.channels}")
        print(f"  Format: 16-bit PCM")
        
        if duration:
            print(f"\n🎤 Recording for {duration} seconds...")
            audio_bytes = recorder.start_recording(duration=duration)
        else:
            print("\n🎤 Recording... Press Enter to stop")
            # Start recording in background thread
            import threading
            recording_thread = threading.Thread(
                target=recorder.start_recording,
                kwargs={'duration': None}
            )
            recording_thread.start()
            
            # Wait for user to press Enter
            input()
            recorder.stop_recording()
            recording_thread.join(timeout=2)
            
            audio_bytes = recorder.get_audio_bytes()
        
        print(f"✅ Recording complete: {len(audio_bytes)} bytes")
        
        # Save recording
        if audio_bytes:
            filename = recorder.save_to_wav()
            return audio_bytes, filename
        else:
            print("❌ No audio recorded")
            return b"", None
            
    except Exception as e:
        print(f"❌ Error recording audio: {e}")
        return b"", None
    finally:
        recorder.cleanup()

# Test function
def test_microphone():
    """Test microphone recording"""
    print("Testing microphone...")
    
    # List available devices
    p = pyaudio.PyAudio()
    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"  Device {i}: {dev['name']} (Input Channels: {dev['maxInputChannels']})")
    p.terminate()
    
    # Test recording
    audio_bytes, filename = record_from_microphone(duration=3)
    
    if audio_bytes:
        print(f"\n✅ Microphone test successful!")
        print(f"   Recorded {len(audio_bytes)} bytes")
        print(f"   Saved to: {filename}")
    else:
        print("\n❌ Microphone test failed")

if __name__ == "__main__":
    test_microphone()