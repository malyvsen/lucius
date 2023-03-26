import pickle


with open("segments.pkl", "rb") as segments_file:
    segments = pickle.load(segments_file)
print("\n".join(segment.text for segment in segments))
