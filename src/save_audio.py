import aria.sdk as aria
from faster_whisper import WhisperModel
from jiwer import wer, cer
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import statistics
import time
import wave

from aria_utils import AriaStreamer
from StreamingClientObserver import AudioObserver



# Reference sentences
test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Can you help me with my homework tonight?",
    "It is raining heavily outside, so take an umbrella.",
    "I will call you when I arrive at the station.",
    "Learning a new language takes time and patience.",
    "Please pass me the salt and pepper from the table.",
    "Did you see the movie last night, or was it too late?",
    "I think we should go for a walk before dinner.",
    "Could you remind me to buy groceries after work?"
]


def save_audio():
    audio_streamer = AriaStreamer()
    data_channels = [aria.StreamingDataType.Audio]
    message_size = 1000
    observer = audio_streamer.stream_subscribe(data_channels, AudioObserver(), message_size)

    out_dir = "/home/jruopp/thesis_ws/src/aria_pkg/data/experiments/audio/"
    # out_dir = "/home/joshy/Bachelorthesis/aria_pkg/data/experiments/audio/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sample_rate = 16000
    duration_sec = 6
    num_samples = sample_rate * duration_sec

    print("Please speak 10 sentences, each lasting 6 seconds. Recording starts automatically.")

    for i in range(10):
        print(f"Ready for recording {i+1}/10 ...")
        print("Recording in progress ...")
        audio_buffer = []
        while len(audio_buffer) < num_samples:
            audios_16k, _ = observer.resample_audio_wav()
            if audios_16k is None or len(audios_16k) == 0:
                time.sleep(0.01)
                continue

            # float32 -1..1 → int16
            audios_16k = np.clip(audios_16k, -1, 1)
            audios_16k = (audios_16k * 32767).astype(np.int16)

            audio_buffer.extend(audios_16k.tolist())
            # print(len(audio_buffer), "samples collected")

        audio_np = np.array(audio_buffer[-num_samples:], dtype=np.int16)
        filename = os.path.join(out_dir, f"sentence_{i+1}.wav")
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_np.tobytes())
        print(f"Gespeichert: {filename}")

    print("Saved all 10 recordings")




def add_white_noise(signal, snr_db, seed=None):
    """Add white noise to signal to achieve target SNR (in dB)."""
    if seed is not None:
        np.random.seed(seed)
    sig_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def benchmark_whisper_whitenoise():
    print("Starting experiment...\n")

    whisper_models = [
        "tiny", "tiny.en",
        "small","small.en",
        "base", "base.en", 
        "medium", "medium.en",
        "large-v3"
    ]

    snr_levels = [None, 15, 5, 0]  # None = clean
    path = "/home/jruopp/thesis_ws/src/aria_pkg/data/experiments/audio/"
    clean_dir = os.path.join(path, "clean")
    results_dir = os.path.join(path, "results_noise")
    os.makedirs(results_dir, exist_ok=True)

    # Load clean wavs once
    clean_files = [os.path.join(clean_dir, f"sentence_{i+1}.wav") for i in range(20)]
    if len(clean_files) < 20:
        print("!! Not enough clean files found")
        return
    clean_audio = [(sf.read(f), os.path.basename(f)) for f in clean_files]  # ((sig, sr), fname)

    for model_name in whisper_models:
        print(f"\nTesting Whisper model: {model_name}")
        model = WhisperModel(model_name, device="cuda", compute_type="int8")

        model_wers, model_cers = [], []
        total_words, total_time = 0, 0
        log_lines = [f"==== Results for Whisper model: {model_name} ====\n"]

        for snr in snr_levels:
            folder_name = "clean" if snr is None else f"{snr} dB"
            folder_wers, folder_cers = [], []
            first_iteration_transcripts = []

            for n in range(10):  # 10 iterations
                for (signal, sr), ref in zip(clean_audio, test_sentences * 2):
                    # Apply noise if requested
                    if snr is None:
                        noisy_signal = signal
                    else:
                        noisy_signal = add_white_noise(signal, snr)

                    start = time.time()
                    segments, _ = model.transcribe(noisy_signal, beam_size=5)
                    duration = time.time() - start

                    transcribed = "".join([seg.text for seg in segments]).strip()
                    ref_words = len(ref.split())
                    total_words += ref_words
                    total_time += duration

                    # Metrics
                    wer_score = wer(ref, transcribed)
                    cer_score = cer(ref, transcribed)
                    folder_wers.append(wer_score)
                    folder_cers.append(cer_score)
                    model_wers.append(wer_score)
                    model_cers.append(cer_score)

                    if n == 0:
                        first_iteration_transcripts.append(
                            f"{folder_name}/{ref[:15]}...\n REF: {ref}\n HYP: {transcribed}\n"
                        )

            # Summarize folder
            mean_wer = statistics.mean(folder_wers)
            mean_cer = statistics.mean(folder_cers)
            std_wer = statistics.stdev(folder_wers) if len(folder_wers) > 1 else 0.0
            std_cer = statistics.stdev(folder_cers) if len(folder_cers) > 1 else 0.0

            log_lines.append(f"\n--- Folder: {folder_name} ---\n")
            log_lines.append("First iteration transcripts:\n")
            log_lines.extend(first_iteration_transcripts)
            log_lines.append(f"\nWER ({folder_name}): mean={mean_wer:.3f}, std={std_wer:.3f}")
            log_lines.append(f"CER ({folder_name}): mean={mean_cer:.3f}, std={std_cer:.3f}\n")

        # Global model results
        global_wer = statistics.mean(model_wers)
        global_cer = statistics.mean(model_cers)
        global_wer_std = statistics.stdev(model_wers) if len(model_wers) > 1 else 0.0
        global_cer_std = statistics.stdev(model_cers) if len(model_cers) > 1 else 0.0
        words_per_sec = total_words / total_time if total_time > 0 else 0

        log_lines.append("\n=== Overall Model Results ===\n")
        log_lines.append(f"Global WER: mean={global_wer:.3f}, std={global_wer_std:.3f}")
        log_lines.append(f"Global CER: mean={global_cer:.3f}, std={global_cer_std:.3f}")
        log_lines.append(f"Transcription Rate: {words_per_sec:.2f} words/sec\n")

        # Save log file
        out_file = os.path.join(results_dir, f"{model_name}_results_whitenoise.txt")
        with open(out_file, "w") as f:
            f.write("\n".join(log_lines))

        print(f"Saved results to {out_file}")




def benchmark_whisper():
    print("Starting experiment...\n")

    whisper_models = [
        "tiny", "tiny.en",
        "small","small.en",
        "base", "base.en", 
        "medium", "medium.en",
        "large-v3"
    ]

    audio_samples = ["clean", "60dB", "80dB", "90dB"]
    path = "/home/jruopp/thesis_ws/src/aria_pkg/data/experiments/audio/"
    results_dir = "/home/jruopp/thesis_ws/src/aria_pkg/data/experiments/results/"

    for model_name in whisper_models:
        print(f"\nTesting Whisper model: {model_name}")
        model = WhisperModel(model_name, device="cuda", compute_type="int8")

        model_wers, model_cers = [], []
        total_words, total_time = 0, 0

        # Collect log text for this model
        log_lines = [f"==== Results for Whisper model: {model_name} ====\n"]

        for folder in audio_samples:
            wav_files = [os.path.join(path, folder, f"sentence_{i+1}.wav") for i in range(20)]
            if len(wav_files) < 20:
                log_lines.append(f"!! Skipping {folder} (not enough files)\n")
                continue

            folder_wers, folder_cers = [], []
            first_iteration_transcripts = []

            for n in range(10):  # 10 iterations

                print(f"Iteration {n+1}")
                for i, (audio_path, ref) in enumerate(zip(wav_files, test_sentences * 2)):
                    start = time.time()
                    segments, _ = model.transcribe(audio_path, beam_size=5)
                    duration = time.time() - start

                    transcribed = "".join([seg.text for seg in segments]).strip()
                    ref_words = len(ref.split())
                    total_words += ref_words
                    total_time += duration

                    # Metrics
                    wer_score = wer(ref, transcribed)
                    cer_score = cer(ref, transcribed)
                    folder_wers.append(wer_score)
                    folder_cers.append(cer_score)
                    model_wers.append(wer_score)
                    model_cers.append(cer_score)

                    if n == 0:
                        first_iteration_transcripts.append(
                            f"{os.path.basename(audio_path)}\n REF: {ref}\n HYP: {transcribed}\n"
                        )

            # Log per-folder results
            log_lines.append(f"\n--- Folder: {folder} ---\n")
            log_lines.append("First iteration transcripts:\n")
            log_lines.extend(first_iteration_transcripts)

            mean_wer = statistics.mean(folder_wers)
            mean_cer = statistics.mean(folder_cers)
            std_wer = statistics.stdev(folder_wers) if len(folder_wers) > 1 else 0.0
            std_cer = statistics.stdev(folder_cers) if len(folder_cers) > 1 else 0.0

            log_lines.append(f"\nWER ({folder}): mean={mean_wer:.3f}, std={std_wer:.3f}")
            log_lines.append(f"CER ({folder}): mean={mean_cer:.3f}, std={std_cer:.3f}\n")

        # Global metrics
        global_wer = statistics.mean(model_wers)
        global_cer = statistics.mean(model_cers)
        global_wer_std = statistics.stdev(model_wers) if len(model_wers) > 1 else 0.0
        global_cer_std = statistics.stdev(model_cers) if len(model_cers) > 1 else 0.0
        words_per_sec = total_words / total_time if total_time > 0 else 0

        log_lines.append("\n=== Overall Model Results ===\n")
        log_lines.append(f"Global WER: mean={global_wer:.3f}, std={global_wer_std:.3f}")
        log_lines.append(f"Global CER: mean={global_cer:.3f}, std={global_cer_std:.3f}")
        log_lines.append(f"Transcription Rate: {words_per_sec:.2f} words/sec\n")

        # Save log file
        out_file = os.path.join(results_dir, f"{model_name}_results.txt")
        with open(out_file, "w") as f:
            f.write("\n".join(log_lines))

        print(f"Saved results to {out_file}")




# 1) Bar chart mit Transkriptionsrate (words/sec) -----------------------
def plot_transcription_rate(models, rates, errors=None):
    """
    models : list of str      -> Modellnamen, z.B. ["tiny", "tiny.en", "small", "small.en", ...]
    rates  : list of float    -> gemessene Transkriptionsraten
    errors : list of float    -> optionale Fehlerbalken (std)
    """

    models = [
        "tiny", "tiny.en",
        "small","small.en",
        "base", "base.en", 
        "medium", "medium.en",
        "large-v3"
    ]

    rates = [142.35, 162.25, 133.61, 157.76, 107.79,169.05, 69.15, 171.08, 47.81]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x, rates, yerr=errors, capsize=5, alpha=0.7)
    ax.plot(x, rates, "ko-", lw=1.5)  # Linie durch die Balken

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Transcription Rate (words/sec)")
    ax.set_title("Whisper Models – Transcription Rate")
    plt.tight_layout()
    plt.show()





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
    # aria = AriaStreamer()
    # device = aria.stream_start(None, "usb", "profile18")

    # save_audio()
    # benchmark_whisper()

    plot_transcription_rate(None, None, errors=None)


if __name__ == "__main__":
    main()
