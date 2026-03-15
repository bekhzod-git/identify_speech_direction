import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from scipy.fft import rfft, rfftfreq

# -----------------------------
# Audio input functions
# -----------------------------
def use_mic(duration=2, samplerate=16000):
    """Record audio from microphone"""
    print("Говорите в микрофон...")
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=1)
    sd.wait()
    return audio.flatten(), samplerate

def use_file(filename="input_audio.wav"):
    """Load audio from file"""
    audio, samplerate = librosa.load(filename, sr=None)
    return audio, samplerate

def save_audio(audio, samplerate, filename="input_audio.wav"):
    """Save audio to WAV file"""
    sf.write(filename, audio, samplerate)
    print(f"Аудио сохранено в файл: {filename}")

# -----------------------------
# Audio metrics functions
# -----------------------------
def rms(audio):
    """Root Mean Square — громкость сигнала.
    Высокое значение → человек близко к микрофону
    Низкое значение → человек далеко / говорит в сторону
    """
    return np.sqrt(np.mean(audio**2))

def high_freq_ratio(audio, samplerate):
    """Доля энергии выше 4000 Hz
    Высокое значение → много высоких частот → говорит прямо
    Низкое значение → звук приглушен → говорит в сторону
    """
    fft_vals = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1/samplerate)
    total_energy = np.sum(fft_vals)
    high_energy = np.sum(fft_vals[freqs > 4000])
    return high_energy / total_energy

def spectral_centroid(audio, samplerate):
    """Центр тяжести спектра.
    Высокий → яркий звук → говорит прямо
    Низкий → приглушенный звук → говорит в сторону
    """
    return librosa.feature.spectral_centroid(y=audio, sr=samplerate).mean()

def spectral_rolloff(audio, samplerate):
    """Частота, ниже которой находится 85% энергии.
    Высокий → много высоких частот → говорит прямо
    Низкий → высокие частоты потеряны → говорит в сторону
    """
    return librosa.feature.spectral_rolloff(y=audio, sr=samplerate).mean()

def spectral_bandwidth(audio, samplerate):
    """Ширина спектра.
    Высокий → широкий спектр → говорит прямо
    Низкий → узкий спектр → говорит в сторону
    """
    return librosa.feature.spectral_bandwidth(y=audio, sr=samplerate).mean()

def spectral_flatness(audio):
    """Показывает насколько спектр похож на шум.
    Низкий → чистый голос → говорит прямо
    Высокий → много шума / реверберации → говорит в сторону
    """
    return librosa.feature.spectral_flatness(y=audio).mean()

def zero_crossing_rate(audio):
    """Количество пересечений нулевой линии.
    Низкий → чистая речь → говорит прямо
    Высокий → шум / отражения → говорит в сторону
    """
    return librosa.feature.zero_crossing_rate(audio).mean()

def mfcc_mean(audio, samplerate, n_mfcc=13):
    """MFCC — форма спектра речи.
    Изменяется при изменении акустики (угол, расстояние)"""
    mfcc = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)

# -----------------------------
# Heuristic decision
# -----------------------------
def heuristic_decision(metrics):
    """Simple scoring system for direct vs off-axis speech"""
    score = 0
    if metrics['rms'] > 0.02:
        score += 1
    if metrics['spectral_centroid'] > 2000:
        score += 1
    if metrics['spectral_rolloff'] > 4000:
        score += 1
    if metrics['spectral_bandwidth'] > 1500:
        score += 1
    if metrics['high_freq_ratio'] > 0.15:
        score += 1
    if score >= 3:
        return "Вероятно человек говорит прямо в микрофон"
    else:
        return "Вероятно человек говорит в сторону или далеко"

# -----------------------------
# Main function
# -----------------------------
def main(get_speech_from='file', filename="input_audio.wav"):
    if get_speech_from == "mic":
        audio, samplerate = use_mic()
    elif get_speech_from == "file":
        audio, samplerate = use_file(filename)
    else:
        raise ValueError(f"No such option: {get_speech_from}")

    save_audio(audio, samplerate)

    # Calculate metrics
    metrics = {
        'rms': rms(audio),
        'high_freq_ratio': high_freq_ratio(audio, samplerate),
        'spectral_centroid': spectral_centroid(audio, samplerate),
        'spectral_rolloff': spectral_rolloff(audio, samplerate),
        'spectral_bandwidth': spectral_bandwidth(audio, samplerate),
        'spectral_flatness': spectral_flatness(audio),
        'zero_crossing_rate': zero_crossing_rate(audio),
        'mfcc': mfcc_mean(audio, samplerate)
    }

    # Print metrics
    print("\nМетрики аудио:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Make heuristic decision
    decision = heuristic_decision(metrics)
    print("\nРезультат:", decision)

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    try:
        main(get_speech_from='file', filename='input_audio.wav')
    except KeyboardInterrupt:
        print("The app is stopped")
    except Exception as e:
        print(f"Error occurred: {e}")

