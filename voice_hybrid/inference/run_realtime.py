# inference/run_realtime.py
import torch
from recorder import record
from utils import load_audio
import numpy as np
import speech_recognition as sr
import librosa

# load models (modify paths as needed)
from audio_emotion.models import CRNN
from text_emotion.models import TextBiLSTM
from fusion.fusion_model import FusionHead

# adjust sizes consistent with training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load audio model (must be saved earlier)
audio_model = CRNN().to(device)
audio_model.load_state_dict(torch.load("checkpoints/crnn_epoch1.pth", map_location=device))
audio_model.eval()

# load text model
# We need vocabulary - load dataset to get vocab
from text_emotion.dataset import TextEmotionDataset
ds = TextEmotionDataset("../data/text/goemotions/train.tsv")
text_model = TextBiLSTM(vocab_size=len(ds.vocab.word2idx), embed_dim=128, hidden=128, n_classes=27).to(device)
text_model.load_state_dict(torch.load("checkpoints/text_bilstm.pth", map_location=device))
text_model.eval()

# simple fusion head (if trained)
fusion = FusionHead(audio_dim=256, text_dim=256, n_classes=7).to(device)
#fusion.load_state_dict(torch.load("checkpoints/fusion.pth"))

def asr_google(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
    except Exception as e:
        text = ""
    return text

def extract_audio_embedding(model, wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    # same preprocessing as training
    import numpy as np
    max_len = int(16000*3.0)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max(0, max_len - len(y))), mode='constant')
    spec = librosa.feature.melspectrogram(y, sr=16000, n_mels=128)
    log_mel = librosa.power_to_db(spec)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std()+1e-9)
    import torch
    x = torch.tensor(log_mel, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        # modify CRNN to return penultimate embedding if you implemented that; otherwise use logits
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

def extract_text_embedding(model, text, ds):
    enc = ds.vocab.encode(text)
    maxlen = 64
    if len(enc) < maxlen:
        enc = enc + [0]*(maxlen - len(enc))
    else:
        enc = enc[:maxlen]
    import torch
    x = torch.tensor([enc], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

if __name__ == "__main__":
    wav = record(duration=3, out_file="tmp.wav")
    asr_text = asr_google(wav)
    print("ASR text:", asr_text)
    a_prob = extract_audio_embedding(audio_model, wav)
    t_prob = extract_text_embedding(text_model, asr_text, ds)
    print("Audio probs shape:", a_prob.shape, "Text probs shape:", t_prob.shape)
    # Simple fusion: average probability across matching label sets (needs mapping)
    # Here we simply show both predictions
    ap = a_prob.squeeze()
    tp = t_prob.squeeze()
    print("Audio top:", ap.argmax(), ap.max())
    print("Text top:", tp.argmax(), tp.max())
