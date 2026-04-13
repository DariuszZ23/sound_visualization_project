# 🎵 Sound Visualization Project

This project is used to visualize audio files using the librosa and matplotlib libraries.

## ✨ Features

The program:

* loads an MP3 audio file provided by the user
* generates a waveform (signal shape)
* generates a spectrogram
* saves plots as PNG files
* displays a simple loading animation in the terminal during processing
* uses multithreading to run the spinner and audio processing simultaneously
* handles the case when the audio file is not found

---

## ⚙️ Requirements

To run the project you need Python 3.10+ and the following libraries:

```bash
pip install librosa matplotlib numpy soundfile audioread
```

---

## 📁 Project Structure

```text
sound_visualization_project/
│
├── main.py
├── waveform.png
├── spectrogram.png
└── README.md
```

---

## ▶️ How to Run

1. Run the program:

```bash
python main.py
```

2. Enter the file name when prompted:

```text
Enter MP3 file name: for example C:\Users\UserName\Music\music.mp3
```

---

## 🔄 How It Works

The program uses multithreading:

* the main thread loads the file and generates plots
* a secondary thread displays a loading animation in the terminal

During execution, a simple animation is shown:

```text
Generating plots... |
Generating plots... /
Generating plots... -
Generating plots... \
```

After processing is complete, you will see:

```text
Done!
```

---

## 💾 Output Files

The program automatically saves generated plots as:

* waveform.png
* spectrogram.png

---

## 📊 Example Output
![waveform.png](waveform.png)
```text
Waveform shows how the amplitude of the sound changes over time. Larger peaks indicate louder parts, while smaller ones indicate quieter sections.
```
![spectrogram.png](spectrogram.png)
```text
Spectrogram shows how frequencies are distributed over time. Brighter colors indicate higher intensity of a frequency, while darker colors indicate lower intensity. Low frequencies are at the bottom, high frequencies at the top.
```
---

## ❌ Error Handling

### FileNotFoundError

If the audio file is not found:

```text
Error: audio file not found.
```

The program stops the spinner, terminates execution, and exits.

---

## 🚀 Possible Extensions

* mel-spectrogram
* chromagram
* beat detection
* MFCC
* animated plots
* GUI using Tkinter
* multiple file support
* file selection via user interface
