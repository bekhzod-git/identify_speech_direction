import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from scipy.fft import rfft, rfftfreq


def use_mic():
    samplerate = 16000
    duration = 2  # длительность записи

    print("Говорите в микрофон...")

    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=1)

    sd.wait()

    audio = audio.flatten()

    return audio, samplerate

def use_file():
    # -----------------------------
    # Load your audio file
    # -----------------------------
    filename = "input_audio.wav"  # change to your file
    audio, samplerate = librosa.load(filename, sr=None)  # keep original sample rate

    return audio, samplerate


def main(get_speech_from: str):
    if get_speech_from == "mic":
        audio, samplerate = use_mic()
    elif get_speech_from == "file":
        audio, samplerate = use_file()
    else:
        raise f"No such option: {get_speech_from}"
    # -------------------------
    # Сохранение аудио
    # -------------------------
    output_filename = "input_audio.wav"
    sf.write(output_filename, audio, samplerate)
    print(f"Аудио сохранено в файл: {output_filename}")

    # =========================================================
    # RMS (Root Mean Square) — громкость сигнала
    # =========================================================
    # Показывает общую мощность аудио сигнала.
    #
    # Интерпретация:
    # высокое значение → человек близко к микрофону
    # низкое значение → человек далеко или отвернулся

    rms = np.sqrt(np.mean(audio**2))


    # =========================================================
    # FFT анализ спектра
    # =========================================================
    # Используется для анализа распределения энергии по частотам

    fft_vals = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1/samplerate)

    total_energy = np.sum(fft_vals)
    high_freq_energy = np.sum(fft_vals[freqs > 4000])

    # =========================================================
    # High Frequency Ratio
    # =========================================================
    # Доля энергии высоких частот (>4000 Hz).
    #
    # Когда человек говорит прямо:
    # микрофон хорошо ловит высокие частоты
    #
    # Интерпретация:
    # высокое значение → человек говорит прямо
    # низкое значение → человек говорит в сторону / далеко

    high_freq_ratio = high_freq_energy / total_energy


    # =========================================================
    # Spectral Centroid
    # =========================================================
    # Центр "тяжести" спектра.
    # Показывает где находится средняя частота сигнала.
    #
    # Если звук яркий (много высоких частот) → centroid высокий
    #
    # Интерпретация:
    # высокий centroid → человек говорит прямо
    # низкий centroid → звук приглушен → человек отвернулся

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=samplerate
    ).mean()


    # =========================================================
    # Spectral Rolloff
    # =========================================================
    # Частота ниже которой находится ~85% энергии сигнала.
    #
    # Если высокие частоты пропадают (человек отвернулся),
    # rolloff становится ниже.
    #
    # Интерпретация:
    # высокий rolloff → много высоких частот → говорит прямо
    # низкий rolloff → высокие частоты потеряны → говорит в сторону

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=samplerate
    ).mean()


    # =========================================================
    # Spectral Bandwidth
    # =========================================================
    # Показывает насколько широкий спектр звука.
    #
    # Если человек говорит прямо:
    # спектр шире (есть и низкие и высокие частоты)
    #
    # Интерпретация:
    # высокий bandwidth → широкий спектр → говорит прямо
    # низкий bandwidth → спектр узкий → говорит в сторону

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=samplerate
    ).mean()


    # =========================================================
    # Spectral Flatness
    # =========================================================
    # Показывает насколько спектр похож на шум.
    #
    # Голос обычно гармонический (не шум).
    #
    # Если появляется много отражений и шума:
    # flatness увеличивается.
    #
    # Интерпретация:
    # низкий flatness → чистый голос → говорит прямо
    # высокий flatness → больше шума / реверберации → говорит в сторону

    flatness = librosa.feature.spectral_flatness(
        y=audio
    ).mean()


    # =========================================================
    # Zero Crossing Rate (ZCR)
    # =========================================================
    # Количество пересечений нулевой линии.
    #
    # У шума и искаженных сигналов ZCR выше.
    #
    # Интерпретация:
    # низкий ZCR → чистая речь → говорит прямо
    # высокий ZCR → шум / отражения → говорит в сторону

    zcr = librosa.feature.zero_crossing_rate(
        audio
    ).mean()


    # =========================================================
    # MFCC (Mel Frequency Cepstral Coefficients)
    # =========================================================
    # Очень важные признаки из обработки речи.
    # Используются почти во всех системах распознавания речи.
    #
    # Они описывают форму спектра речи.
    # Изменяются если меняется акустика (угол, расстояние).

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=samplerate,
        n_mfcc=13
    )

    mfcc_mean = mfcc.mean(axis=1)


    # =========================================================
    # Вывод метрик
    # =========================================================

    print("\nМетрики аудио:")

    print("RMS:", rms)
    print("High freq ratio:", high_freq_ratio)
    print("Spectral centroid:", centroid)
    print("Spectral rolloff:", rolloff)
    print("Spectral bandwidth:", bandwidth)
    print("Spectral flatness:", flatness)
    print("Zero Crossing Rate:", zcr)

    print("\nMFCC:")
    print(mfcc_mean)


    # =========================================================
    # Простая эвристическая классификация
    # =========================================================

    score = 0

    if rms > 0.02:
        score += 1

    if centroid > 2000:
        score += 1

    if rolloff > 4000:
        score += 1

    if bandwidth > 1500:
        score += 1

    if high_freq_ratio > 0.15:
        score += 1

    if score >= 3:
        print("\nВероятно человек говорит прямо в микрофон")
    else:
        print("\nВероятно человек говорит в сторону или далеко")


if __name__ == "__main__":
    try:
        main(get_speech_from='file')
    except KeyboardInterrupt:
        print(f"The app is stopped")
    except Exception as e:
        print(f"Error occurred: {e}")

