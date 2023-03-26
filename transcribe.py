import pickle
from tqdm import tqdm
from faster_whisper import WhisperModel


model = WhisperModel("medium", device="cpu", compute_type="int8")
segments, info = model.transcribe("/home/malyvsen/temp/lecture-11.mp3")

with open("segments.pkl", "wb") as segments_file:
    pickle.dump(
        [segment for segment in tqdm(segments, desc="Transcribing")], segments_file
    )
