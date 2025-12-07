"""
Example usage of Groq audio adapters.

This demonstrates:
1. Text-to-Speech (TTS) with multiple voices
2. Audio transcription (Whisper)
3. Audio translation (Whisper)
"""

import os
from jugglerr import (
    GroqTTSAdapter,
    GroqTranscriptionAdapter,
    GroqTranslationAdapter,
    groq_speak,
    groq_transcribe,
    groq_translate
)

# =============================================================================
# Example 1: Text-to-Speech (TTS)
# =============================================================================

print("Example 1: Text-to-Speech")
print("-" * 60)

# Using the adapter class
tts_adapter = GroqTTSAdapter()

# Generate English speech
response = tts_adapter.generate_speech(
    text="Welcome to Groq's text to speech API!",
    voice="Fritz-PlayAI",
    model="playai-tts",
    response_format="wav"
)

# Save to file
response.write_to_file("output_english.wav")
print(f"✓ Generated English speech: output_english.wav")

# Generate Arabic speech
arabic_response = tts_adapter.generate_speech(
    text="مرحبا بك في واجهة برمجة التطبيقات",
    voice="Ahmad-PlayAI",
    model="playai-tts-arabic"
)

arabic_response.write_to_file("output_arabic.wav")
print(f"✓ Generated Arabic speech: output_arabic.wav")

# List available voices
english_voices = tts_adapter.list_voices("playai-tts")
arabic_voices = tts_adapter.list_voices("playai-tts-arabic")
print(f"✓ Available English voices: {len(english_voices)}")
print(f"✓ Available Arabic voices: {len(arabic_voices)}")

# =============================================================================
# Example 2: Quick TTS with convenience function
# =============================================================================

print("\nExample 2: Quick TTS")
print("-" * 60)

# One-liner to generate and save speech
groq_speak(
    "This is a quick example!",
    voice="Celeste-PlayAI",
    output_file="quick_speech.wav"
)
print("✓ Generated quick speech: quick_speech.wav")

# =============================================================================
# Example 3: Audio Transcription (requires audio file)
# =============================================================================

print("\nExample 3: Audio Transcription")
print("-" * 60)

# Uncomment if you have an audio file to transcribe
"""
transcription_adapter = GroqTranscriptionAdapter()

# Transcribe audio file
response = transcription_adapter.transcribe(
    file="path/to/audio.mp3",
    model="whisper-large-v3",
    language="en",  # Optional: improves accuracy
    response_format="verbose_json"  # Get detailed info
)

print(f"Transcription: {response.text}")
print(f"Language: {response.language}")
print(f"Duration: {response.duration}s")

# Quick transcription
text = groq_transcribe("path/to/audio.mp3", language="en")
print(f"Quick transcription: {text}")
"""

# =============================================================================
# Example 4: Audio Translation (requires audio file)
# =============================================================================

print("\nExample 4: Audio Translation")
print("-" * 60)

# Uncomment if you have a non-English audio file
"""
translation_adapter = GroqTranslationAdapter()

# Translate audio to English
response = translation_adapter.translate(
    file="path/to/spanish_audio.mp3",
    model="whisper-large-v3-turbo"  # Faster model
)

print(f"Translation (English): {response.text}")

# Quick translation
english_text = groq_translate("path/to/spanish_audio.mp3")
print(f"Quick translation: {english_text}")
"""

# =============================================================================
# Example 5: Different TTS voices
# =============================================================================

print("\nExample 5: Trying Different Voices")
print("-" * 60)

sample_text = "The quick brown fox jumps over the lazy dog."

# Try a few different voices
voices_to_try = ["Fritz-PlayAI", "Celeste-PlayAI", "Thunder-PlayAI"]

for voice in voices_to_try:
    filename = f"voice_{voice.replace('-PlayAI', '').lower()}.wav"
    groq_speak(sample_text, voice=voice, output_file=filename)
    print(f"✓ Generated with {voice}: {filename}")

print("\n✅ All examples complete!")
print("\nGenerated files:")
print("  - output_english.wav")
print("  - output_arabic.wav")
print("  - quick_speech.wav")
print("  - voice_fritz.wav")
print("  - voice_celeste.wav")
print("  - voice_thunder.wav")
