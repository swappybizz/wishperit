import sys,os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit
import pyaudio
import wave
import threading
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import QTimer, QDateTime, QObject, pyqtSignal, QThread
import pyaudio
import wave
import threading

class AudioRecorderPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.is_playing = False
        self.record_duration = 5000  # milliseconds
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.stopRecording)

    def initUI(self):
        layout = QVBoxLayout()

        self.status_label = QLabel('Ready', self)
        layout.addWidget(self.status_label)

        self.record_button = QPushButton('Record', self)
        self.record_button.clicked.connect(self.toggleRecording)
        layout.addWidget(self.record_button)

        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.playAudio)
        
        transcript_button = QPushButton('Transcript', self)
        transcript_button.clicked.connect(self.transcript)
        
        self.transcription_area = QTextEdit("Transcription goes here", self)
        layout.addWidget(self.play_button)
        layout.addWidget(transcript_button)
        layout.addWidget(self.transcription_area)

        self.setLayout(layout)
        self.setWindowTitle('Audio Recorder & Player')
        self.setGeometry(300, 300, 300, 200)
        
    def transcript(self):
        # Start the transcription in a new thread
        self.thread = QThread()
        self.worker = TranscriptionWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.transcribe)
        self.worker.finished.connect(self.updateTranscription)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        
    def updateTranscription(self, text):
        # Update the UI with the transcription result
        self.transcription_area.setText(text)


    def toggleRecording(self):
        if self.is_recording:
            self.stopRecording()
        else:
            self.is_recording = True
            self.record_button.setText('Stop')
            self.status_label.setText('Recording...')
            self.frames = []
            threading.Thread(target=self.recordAudio).start()
            # self.timer.start(self.record_duration)

    def recordAudio(self):
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        while self.is_recording:
            data = self.stream.read(1024)
            self.frames.append(data)
        self.stream.stop_stream()
        self.stream.close()
        self.saveRecording()

    def stopRecording(self):
        self.is_recording = False
        self.record_button.setText('Record')
        self.status_label.setText('Ready')
        # self.timer.stop()

    def saveRecording(self):
        current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
        self.filename = f"recording_{current_time}.wav"  # Update self.filename with the new recording
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.status_label.setText(f'Recording saved: {self.filename}')

    def playAudio(self):
        if self.is_playing or self.is_recording:
            return
        self.status_label.setText('Playing...')
        self.is_playing = True
        threading.Thread(target=self.playback).start()

    def playback(self):
        try:
            wf = wave.open(self.filename, 'rb')
            stream = self.audio.open(format=self.audio.get_format_from_width(wf.getsampwidth()),
                                    channels=wf.getnchannels(),
                                    rate=wf.getframerate(),
                                    output=True)
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            wf.close()
        except FileNotFoundError:
            self.status_label.setText('File not found. Please record something first.')
        except Exception as e:
            self.status_label.setText(f'Error playing the file: {e}')
        self.is_playing = False
        # on finish playing audio write the status to ready
        self.status_label.setText('Ready')
        
class TranscriptionWorker(QObject):
    finished = pyqtSignal(str)  # Signal to return the transcription result

    def transcribe(self):
        # Define the model directory
        model_directory = os.path.join('.', 'models')
        model_id = "openai/whisper-base"

        # Check and set device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Ensure the models directory exists
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Load or download the model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=model_directory  # Specify where to look for and save the model
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_directory)

        # Set up the pipeline with the model and processor
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Find the latest .wav file in the current directory
        files = [f for f in os.listdir('.') if f.endswith('.wav')]
        latest_file = max(files, key=os.path.getctime)

        # Transcribe the latest .wav file
        result = pipe(latest_file)
        text = result['text']

        # Emit the finished signal with the transcription result
        self.finished.emit(text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioRecorderPlayer()
    ex.show()
    sys.exit(app.exec())
