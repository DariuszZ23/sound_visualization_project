import itertools
import sys
import threading
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time

loading = False

def spinner():
    global loading
    for char in itertools.cycle(['|', '/', '-', '\\']):
        if not loading:
            break
        sys.stdout.write(f'\rGenerowanie wykresów... {char}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rGotowe!                    \n')

def visualize_sound(name):
    mp3_file_name = input("Podaj nazwę pliku MP3: ")
    global loading
    # Uruchom spinner
    loading = True
    t = threading.Thread(target=spinner)
    t.start()

    try:
        y, sr = librosa.load(mp3_file_name, sr=None)
    except FileNotFoundError:
        loading = False
        t.join()
        print("\rBłąd: nie znaleziono pliku audio.")
        sys.exit()

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

    plt.show()
    # Zatrzymaj spinner
    loading = False
    t.join()



if __name__ == '__main__':
    visualize_sound('PyCharm')
