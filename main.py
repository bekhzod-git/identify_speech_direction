import sounddevice as sd
import numpy as np
from scipy.fft import rfft, rfftfreq

samplerate = 16000
duration = 2  # секунды записи

print("Говорите в микрофон...")

audio = sd.rec(int(duration * samplerate),
               samplerate=samplerate,
               channels=1)

sd.wait()

audio = audio.flatten()

# RMS громкость
rms = np.sqrt(np.mean(audio**2))

# FFT спектр
fft_vals = np.abs(rfft(audio))
freqs = rfftfreq(len(audio), 1/samplerate)

# энергия высоких частот (>4000 Hz)
high_freq_energy = np.sum(fft_vals[freqs > 4000])
total_energy = np.sum(fft_vals)

ratio = high_freq_energy / total_energy

print("RMS:", rms)
print("High freq ratio:", ratio)

# простая эвристика
if rms > 0.02 and ratio > 0.15:
    print("Человек говорит прямо в микрофон")
else:
    print("Вероятно человек говорит в сторону или далеко")