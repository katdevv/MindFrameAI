import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

speech_config = speechsdk.SpeechConfig(
    subscription=os.environ["AZURE_SPEECH_KEY"],
    region=os.environ["AZURE_SPEECH_REGION"]
)

# Use SSML for more explicit voice control
ssml_text = """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ka-GE">
    <voice name="ka-GE-GiorgiNeural">
        გამარჯობა, ეს არის ტესტი ქართულ ენაზე.
    </voice>
</speak>
"""

# Always use file output to avoid audio system issues
audio_config = speechsdk.audio.AudioOutputConfig(filename="georgian_ssml.wav")
synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config, 
    audio_config=audio_config
)

print("Attempting synthesis with SSML...")
result = synthesizer.speak_ssml_async(ssml_text).get()

if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("✅ SSML synthesis successful!")
    print("Generated: georgian_ssml.wav")
elif result.reason == speechsdk.ResultReason.Canceled:
    details = speechsdk.CancellationDetails.from_result(result)
    print(f"❌ SSML synthesis failed: {details.reason}")
    if details.reason == speechsdk.CancellationReason.Error:
        print(f"Error details: {details.error_details}")