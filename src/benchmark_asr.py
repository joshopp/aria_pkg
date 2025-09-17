import argparse
import aria.sdk as aria
from faster_whisper import WhisperModel
from jiwer import wer, cer
import numpy as np
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record_audio",
        default=False,
        action="store_true",
        help="lets the user record 10 test sentences a 6 seconds",
    )
    parser.add_argument(
        "--benchmark_whisper",
        default=False,
        action="store_true",
        help="activates benchmarking of the stored .wav files",
    )
    return parser.parse_args()



def save_audio(filepath):
    audio_streamer = AriaStreamer()
    data_channels = [aria.StreamingDataType.Audio]
    message_size = 1000
    observer = audio_streamer.stream_subscribe(data_channels, AudioObserver(), message_size)

    out_dir = os.path.join(filepath, "data/experiments/audio")
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

            # normalize audios and add to list
            audios_16k = np.clip(audios_16k, -1, 1)
            audios_16k = (audios_16k * 32767).astype(np.int16)
            audio_buffer.extend(audios_16k.tolist())

        # save data as .wav file
        audio_np = np.array(audio_buffer[-num_samples:], dtype=np.int16)
        filename = os.path.join(out_dir, f"sentence_{i+1}.wav")
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_np.tobytes())
        print(f"saved: {filename}")
    print("Saved all 10 recordings")


def benchmark_whisper(filepath):
    print("Starting benchmarking...\n")

    whisper_models = [
        "tiny", "tiny.en",
        "small","small.en",
        "base", "base.en", 
        "medium", "medium.en",
        "large-v3"
    ]
    audio_samples = ["clean", "60dB", "80dB", "90dB"]

    for model_name in whisper_models:
        print(f"\nTesting Whisper model: {model_name}")
        model = WhisperModel(model_name, device="cuda", compute_type="int8")

        model_wers, model_cers = [], []
        total_words, total_time = 0, 0
        log_lines = [f"==== Results for Whisper model: {model_name} ====\n"]

        # Test data for each model
        for folder in audio_samples:
            # 20 wav files in each folder
            wav_files = [os.path.join(filepath, "data/experiments/audio",  folder, f"sentence_{i+1}.wav") for i in range(20)]
            if len(wav_files) < 20:
                log_lines.append(f"!! Skipping {folder} (not enough files)\n")
                continue

            folder_wers, folder_cers = [], []
            first_iteration_transcripts = []
            for n in range(10):  # 10 iterations
                print(f"Test sentence {n+1}")
                for i, (audio_path, ref) in enumerate(zip(wav_files, test_sentences * 2)):
                    start = time.time()
                    segments, _ = model.transcribe(audio_path, beam_size=5)
                    duration = time.time() - start
                    transcribed = "".join([seg.text for seg in segments]).strip()
                    ref_words = len(ref.split())
                    total_words += ref_words
                    total_time += duration

                    # metrics
                    wer_score = wer(ref, transcribed)
                    cer_score = cer(ref, transcribed)
                    folder_wers.append(wer_score)
                    folder_cers.append(cer_score)
                    model_wers.append(wer_score)
                    model_cers.append(cer_score)

                    # save full transcription results of first iteration
                    if n == 0:
                        first_iteration_transcripts.append(
                            f"{os.path.basename(audio_path)}\n REF: {ref}\n HYP: {transcribed}\n"
                        )

            # log results
            log_lines.append(f"\n--- Folder: {folder} ---\n")
            log_lines.append("First iteration transcripts:\n")
            log_lines.extend(first_iteration_transcripts)

            mean_wer = statistics.mean(folder_wers)
            mean_cer = statistics.mean(folder_cers)
            std_wer = statistics.stdev(folder_wers) if len(folder_wers) > 1 else 0.0
            std_cer = statistics.stdev(folder_cers) if len(folder_cers) > 1 else 0.0

            log_lines.append(f"\nWER ({folder}): mean={mean_wer:.3f}, std={std_wer:.3f}")
            log_lines.append(f"CER ({folder}): mean={mean_cer:.3f}, std={std_cer:.3f}\n")

        # log global results
        global_wer = statistics.mean(model_wers)
        global_cer = statistics.mean(model_cers)
        global_wer_std = statistics.stdev(model_wers) if len(model_wers) > 1 else 0.0
        global_cer_std = statistics.stdev(model_cers) if len(model_cers) > 1 else 0.0
        words_per_sec = total_words / total_time if total_time > 0 else 0

        log_lines.append("\n=== Overall Model Results ===\n")
        log_lines.append(f"Global WER: mean={global_wer:.3f}, std={global_wer_std:.3f}")
        log_lines.append(f"Global CER: mean={global_cer:.3f}, std={global_cer_std:.3f}")
        log_lines.append(f"Transcription Rate: {words_per_sec:.2f} words/sec\n")

        # save log file
        out_file = os.path.join(filepath, "data/experiments/results", f"{model_name}_results.txt")
        with open(out_file, "w") as f:
            f.write("\n".join(log_lines))
        print(f"Saved results to {out_file}")

 
def main():
    srcpath = os.path.dirname(os.path.abspath(__file__))
    filepath = srcpath[:-4]  # Adjust path to project root

    args = parse_args()

    if args.record_audio:
        # start audio streaming 
        aria = AriaStreamer()
        device = aria.stream_start(None, "usb", "profile18")
        # save 10 audio files
        save_audio(filepath)

    if args.benchmark_whisper:
        # benchmark all wav files
        benchmark_whisper(filepath)



if __name__ == "__main__":
    main()