# voice_hybrid

Hybrid audio + text emotion detection (from scratch models).

## Setup
1. Create virtualenv and install:
   pip install -r requirements.txt

2. Prepare data:
   - Put RAVDESS and CREMA-D wavs into data/audio/ravdess and data/audio/creamad
   - Create manifests/manifests using manifests/make_audio_manifest.py
   - Place GoEmotions files under data/text/goemotions/

3. Train audio model:
   python audio_emotion/train_audio.py --manifest ../manifests/audio_manifest.csv --epochs 30 --batch 16

4. Train text model:
   python text_emotion/train_text.py --tsv ../data/text/goemotions/train.tsv

5. (Optional) Train fusion head after preparing a dataset with aligned audio+transcript+label.

6. Run realtime inference (prototype):
   python inference/run_realtime.py
