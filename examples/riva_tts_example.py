"""
NVIDIA Riva TTS Example (Self-Hosted)

This example demonstrates how to use NVIDIA Magpie TTS models
with a self-hosted Riva TTS NIM instance.

Prerequisites:
1. Docker container running: nvcr.io/nim/nvidia/magpie-tts-zeroshot:latest
2. Container accessible at http://localhost:9000
3. NVIDIA AI Enterprise license

Deployment:
docker run -d \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -p 9000:9000 \
  -p 50051:50051 \
  --gpus all \
  nvcr.io/nim/nvidia/magpie-tts-zeroshot:latest
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from jugglerr import RivaTTSAdapter

# Initialize adapter (point to your self-hosted instance)
tts = RivaTTSAdapter(base_url="http://localhost:9000")

print("="*80)
print("NVIDIA RIVA TTS EXAMPLES (SELF-HOSTED)")
print("="*80)

# Example 1: Health Check
print("\n1. Health Check")
print("-" * 80)
if tts.health_check():
    print("✅ TTS service is ready")
else:
    print("❌ TTS service is not ready")
    print("   Make sure Docker container is running on port 9000")
    sys.exit(1)

# Example 2: List Available Voices
print("\n2. List Available Voices")
print("-" * 80)
try:
    voices = tts.list_voices()
    for language, data in voices.items():
        print(f"\n{language}:")
        for voice in data.get('voices', []):
            print(f"  - {voice}")
except Exception as e:
    print(f"Error listing voices: {e}")

# Example 3: Simple Text-to-Speech
print("\n3. Simple Text-to-Speech")
print("-" * 80)
try:
    audio = tts.synthesize(
        text="Hello, this is a test of NVIDIA Riva text to speech.",
        language="en-US"
    )
    
    output_file = "output_simple.wav"
    with open(output_file, "wb") as f:
        f.write(audio)
    
    print(f"✅ Generated audio: {output_file}")
    print(f"   Size: {len(audio)} bytes")
except Exception as e:
    print(f"❌ Error: {e}")

# Example 4: Text-to-Speech with Specific Voice
print("\n4. Text-to-Speech with Specific Voice")
print("-" * 80)
try:
    audio = tts.synthesize(
        text="This is using a specific voice.",
        language="en-US",
        voice="Magpie-Multilingual.EN-US.Aria"  # Use actual voice ID from list_voices
    )
    
    output_file = "output_voice.wav"
    with open(output_file, "wb") as f:
        f.write(audio)
    
    print(f"✅ Generated audio: {output_file}")
except Exception as e:
    print(f"❌ Error: {e}")

# Example 5: Voice Cloning (Zeroshot)
print("\n5. Voice Cloning with Audio Prompt")
print("-" * 80)
print("Note: Requires a 5-second WAV file as voice sample")
print("      Format: Mono, 16-bit, ≥22.05kHz, low noise")

# Uncomment if you have a voice sample
# try:
#     audio = tts.synthesize(
#         text="This is synthesized speech using voice cloning.",
#         language="en-US",
#         audio_prompt_path="voice_sample.wav"
#     )
#     
#     output_file = "output_cloned.wav"
#     with open(output_file, "wb") as f:
#         f.write(audio)
#     
#     print(f"✅ Generated cloned voice audio: {output_file}")
# except Exception as e:
#     print(f"❌ Error: {e}")

# Example 6: Streaming TTS (Zeroshot only)
print("\n6. Streaming Text-to-Speech")
print("-" * 80)
print("Note: Only works with magpie-tts-zeroshot, not magpie-tts-flow")
try:
    chunks = []
    chunk_count = 0
    
    for chunk in tts.synthesize_stream(
        text="This is streaming text to speech synthesis.",
        language="en-US",
        chunk_size=1024
    ):
        chunks.append(chunk)
        chunk_count += 1
    
    # Combine chunks
    audio_data = b''.join(chunks)
    
    print(f"✅ Received {chunk_count} chunks")
    print(f"   Total size: {len(audio_data)} bytes")
    print(f"   Note: This is raw PCM, not WAV format")
    
    # To save as WAV, you'd need to add WAV header
    # For now, just save raw PCM
    output_file = "output_stream.pcm"
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    print(f"   Saved as: {output_file}")
    print(f"   To play: ffplay -f s16le -ar 22050 -ac 1 {output_file}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Example 7: Multilingual TTS
print("\n7. Multilingual Text-to-Speech")
print("-" * 80)
languages = [
    ("en-US", "Hello, how are you today?"),
    ("es-ES", "Hola, ¿cómo estás hoy?"),
    ("fr-FR", "Bonjour, comment allez-vous aujourd'hui?"),
    ("de-DE", "Hallo, wie geht es dir heute?"),
]

for language, text in languages:
    try:
        audio = tts.synthesize(
            text=text,
            language=language,
            sample_rate_hz=22050
        )
        
        output_file = f"output_{language}.wav"
        with open(output_file, "wb") as f:
            f.write(audio)
        
        print(f"✅ {language}: {output_file} ({len(audio)} bytes)")
    except Exception as e:
        print(f"❌ {language}: {e}")

# Example 8: Different Sample Rates
print("\n8. Different Sample Rates")
print("-" * 80)
sample_rates = [16000, 22050, 44100, 48000]

for rate in sample_rates:
    try:
        audio = tts.synthesize(
            text="Testing different sample rates.",
            language="en-US",
            sample_rate_hz=rate
        )
        
        output_file = f"output_{rate}hz.wav"
        with open(output_file, "wb") as f:
            f.write(audio)
        
        print(f"✅ {rate}Hz: {output_file} ({len(audio)} bytes)")
    except Exception as e:
        print(f"❌ {rate}Hz: {e}")

print("\n" + "="*80)
print("EXAMPLES COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - output_simple.wav")
print("  - output_voice.wav")
print("  - output_stream.pcm")
print("  - output_*.wav (multilingual)")
print("  - output_*hz.wav (different sample rates)")
print("\nTo play WAV files:")
print("  - Windows: Use Windows Media Player or VLC")
print("  - Linux/Mac: ffplay output_simple.wav")
print("\nTo play PCM file:")
print("  - ffplay -f s16le -ar 22050 -ac 1 output_stream.pcm")
