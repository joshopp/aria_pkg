import difflib
def benchmark_transcriptions():
    """
    Vergleicht die Transkripte der 7 Audio-Dateien mit den test_sentences und gibt einen Score aus.
    """
    from faster_whisper import WhisperModel
    path = "/home/jruopp/thesis_ws/src/aria_pkg/data/experiments/audio/"
    wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    wav_files.sort()  # Annahme: sentence_1.wav, sentence_2.wav, ...
    if len(wav_files) < 7:
        print("Nicht genug WAV-Dateien gefunden!")
        return
    # Nur die ersten 7 nehmen
    wav_files = wav_files[:7]
    # Whisper-Modell wählen (z.B. base.en)
    model = WhisperModel("base.en", device="cuda", compute_type="int8")
    transcriptions = []
    for wav_file in wav_files:
        audio_path = os.path.join(path, wav_file)
        segments, _ = model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text.strip() for segment in segments])
        transcriptions.append(text)
    print("\nBenchmarking Ergebnisse:")
    total_score = 0
    for i, (ref, hyp) in enumerate(zip(test_sentences, transcriptions)):
        # Ähnlichkeit berechnen (z.B. SequenceMatcher)
        matcher = difflib.SequenceMatcher(None, ref.lower(), hyp.lower())
        score = matcher.ratio()
        total_score += score
        print(f"Satz {i+1}:")
        print(f"  Reference:   {ref}")
        print(f"  Transkribiert: {hyp}")
        print(f"  Ähnlichkeit: {score:.2f}\n")
    print(f"Durchschnittliche Ähnlichkeit: {total_score/7:.2f}")
#! /usr/bin/env python
import os
import numpy as np
import wave
import aria.sdk as aria
from StreamingClientObserver import AudioObserver
from aria_utils import AriaStreamer
from faster_whisper import WhisperModel


def save_audio():
    audio_streamer = AriaStreamer()
    data_channels = [aria.StreamingDataType.Audio]
    message_size = 100
    observer = audio_streamer.stream_subscribe(data_channels, AudioObserver(), message_size)

    out_dir = "/home/jruopp/thesis_ws/src/aria_pkg/data/experiments/audio/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sample_rate = 16000
    duration_sec = 8
    num_samples = sample_rate * duration_sec

    print("Bitte sprich 7 Sätze, jeweils 15 Sekunden. Aufnahme startet automatisch, sobald Audio empfangen wird.")

    for i in range(7):
        print(f"Bereit für Aufnahme {i+1}/7 ...")
        # Warte auf neue Daten
        while not observer.received:
            pass
        print("Aufnahme läuft ...")
        audio_buffer = []
        while len(audio_buffer) < num_samples:
            audios_16k, _ = observer.resample_audio()
            audio_buffer.extend(audios_16k.tolist())
        audio_np = np.array(audio_buffer[-num_samples:], dtype=np.int16)
        filename = os.path.join(out_dir, f"sentence_{i+1}.wav")
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_np.tobytes())
        print(f"Gespeichert: {filename}")
        observer.received = False  # Für nächste Aufnahme zurücksetzen

    print("Alle 7 Aufnahmen gespeichert.")




def test_whisper():
    print("Starting experiment...\n")
    # Modellnamen
    whisper_models = ["small", "small.en", "base", "base.en", 
                      "medium", "medium.en",
                      "large-v2", "large-v2.en"]

    path = "/home/jruopp/thesis_ws/src/aria_pkg/data/experiments/audio/"
    # Alle WAV-Dateien im Verzeichnis finden
    wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    if not wav_files:
        print("Keine WAV-Dateien gefunden!")
        return

    for model_name in whisper_models:
        print(f"\nTesting Whisper model: {model_name}")
        model = WhisperModel(model_name, device="cuda", compute_type="int8")
        for wav_file in wav_files:
            audio_path = os.path.join(path, wav_file)
            print("used audio file: ", audio_path, "\n")
            segments, _ = model.transcribe(audio_path, beam_size=5)
            for segment in segments:
                result = {segment.text}


test_sentences = ["The quick brown fox jumps over the lazy dog.",
                  "She sells seashells by the seashore.",
                  "Can you help me with my homework tonight?",
                  "It is raining heavily outside, so take an umbrella.",
                  "I will call you when I arrive at the station.",
                  "Learning a new language takes time and patience.",
                  "Please pass me the salt and pepper from the table.",
                  "Did you see the movie last night, or was it too late?",
                  "I think we should go for a walk before dinner.",
                  "Could you remind me to buy groceries after work?"]   

def main():
    save_audio()
    # Nach der Aufnahme und Speicherung kannst du benchmark_transcriptions() aufrufen:
    # benchmark_transcriptions()

if __name__ == "__main__":
    main()
