

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_sound(name):

    y, sr = librosa.load("Pufino_Thoughtful(freetouse.com).mp3", sr=None)

    # 1. Waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform dźwięku")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.tight_layout()
    plt.savefig("waveform.png", dpi=300)
    plt.show()

    # 2. Spectrogram
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis='time',
        y_axis='log',
        cmap='magma'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.tight_layout()
    plt.savefig("spectrogram.png", dpi=300)
    plt.show()  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    visualize_sound('PyCharm')
