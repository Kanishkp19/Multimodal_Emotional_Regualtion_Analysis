"""
Emotion Regulation Inference — M4 Mac
======================================
Loads trained checkpoints and computes ECI score for a given input.

Usage:
    python infer.py --video path/to/video.mp4 --audio path/to/audio.wav --text "your transcript"
    python infer.py --audio path/to/audio.wav --text "your transcript"          # no video
    python infer.py --text "your transcript"                                      # text only
"""

import argparse
import numpy as np
import torch
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'
INC_CKPT  = CHECKPOINT_DIR / 'incongruence_encoder_best.pth'
TEMP_CKPT = CHECKPOINT_DIR / 'temporal_model_best.pth'


# ── Model definitions (must match training) ──────────────────────────────────
import torch.nn as nn
import torch.nn.functional as F

class IncongruenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512,256,128], output_dim=64, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, face, audio, text):
        return F.normalize(self.encoder(torch.cat([face, audio, text], dim=1)), p=2, dim=1)


class TemporalModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True, batch_first=True)
        d = hidden_dim * 2
        self.attn = nn.Sequential(nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1))
        self.fc   = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d//2, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        scores   = self.attn(out)
        attended = torch.sum(out * torch.softmax(scores, dim=1), dim=1)
        return self.fc(attended)


# ── Feature extractors ───────────────────────────────────────────────────────
def extract_face(video_path: str, skip_frames: int = 5):
    """Extract MediaPipe face features from video. Returns (T, 944) or None."""
    try:
        import cv2
        import mediapipe as mp

        mp_fm = mp.solutions.face_mesh
        face_mesh = mp_fm.FaceMesh(static_image_mode=False, max_num_faces=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(video_path)
        seq, idx = [], 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if idx % skip_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    fl = res.multi_face_landmarks[0]
                    lm = [v for l in fl.landmark for v in [l.x, l.y]]
                    lms = [(l.x, l.y, l.z) for l in fl.landmark]
                    ex = [
                        np.linalg.norm(np.array(lms[33]) - np.array(lms[160])),
                        np.linalg.norm(np.array(lms[362]) - np.array(lms[385])),
                        np.linalg.norm(np.array(lms[13]) - np.array(lms[14])),
                        np.linalg.norm(np.array(lms[61]) - np.array(lms[291])),
                        lms[70][2], lms[300][2],
                        abs(lms[234][2] - lms[454][2]),
                        np.linalg.norm(np.array(lms[152]) - np.array(lms[10]))
                    ]
                    seq.append(np.concatenate([lm, ex]))
            idx += 1

        cap.release()
        face_mesh.close()
        return np.array(seq) if seq else None
    except Exception as e:
        print(f'  Warning: face extraction failed ({e}) — using zeros')
        return None


def extract_audio(audio_path: str):
    """Extract MFCC + prosody features. Returns (T, D) array."""
    import librosa
    import warnings; warnings.filterwarnings('ignore')

    y, sr = librosa.load(audio_path, sr=16000)
    hop = 512
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=hop)
    d1    = librosa.feature.delta(mfcc)
    d2    = librosa.feature.delta(mfcc, order=2)
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=2093, sr=sr, hop_length=hop)
    f0    = np.nan_to_num(f0, nan=0.0)
    rms   = librosa.feature.rms(y=y, hop_length=hop)[0]
    zcr   = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    cent  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    roll  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    bw    = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    flat  = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    return np.vstack([mfcc, d1, d2, f0.reshape(1,-1), rms.reshape(1,-1),
                      zcr.reshape(1,-1), cent.reshape(1,-1), roll.reshape(1,-1),
                      bw.reshape(1,-1), flat.reshape(1,-1)]).T


def extract_text(text: str):
    """Extract RoBERTa + GoEmotions features. Returns (796,) array."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import warnings; warnings.filterwarnings('ignore')

    model_name = 'SamLowe/roberta-base-go_emotions'
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    inp = tok(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    with torch.no_grad():
        out    = model(**inp, output_hidden_states=True)
        emb    = out.hidden_states[-1][:, 0, :].numpy()[0]   # (768,)
        logits = out.logits.numpy()[0]                        # (28,)
    return np.concatenate([emb, logits])                      # (796,)


# ── Main inference ───────────────────────────────────────────────────────────
def run_inference(video_path=None, audio_path=None, text=None):
    # Load checkpoints
    if not INC_CKPT.exists() or not TEMP_CKPT.exists():
        print(f'ERROR: Checkpoints not found in {CHECKPOINT_DIR}')
        print('Make sure incongruence_encoder_best.pth and temporal_model_best.pth are in ./checkpoints/')
        sys.exit(1)

    inc_ckpt  = torch.load(INC_CKPT,  map_location='cpu')
    temp_ckpt = torch.load(TEMP_CKPT, map_location='cpu')
    input_dim = inc_ckpt.get('input_dim', 1786)

    inc  = IncongruenceEncoder(input_dim=input_dim)
    temp = TemporalModel()
    inc.load_state_dict(inc_ckpt['model_state_dict'])
    temp.load_state_dict(temp_ckpt['model_state_dict'])
    inc.eval(); temp.eval()
    print(f'✓ Models loaded  (input_dim={input_dim})')

    # ── Extract features ─────────────────────────────────────────────────────
    print('\nExtracting features...')
    FACE_DIM  = 944
    AUDIO_DIM = input_dim - FACE_DIM - 796

    # Face
    if video_path:
        print(f'  Face  ← {video_path}')
        face_seq = extract_face(video_path)
    else:
        face_seq = None

    if face_seq is None:
        print('  Face  → using zeros (no video or detection failed)')
        face_seq = np.zeros((10, FACE_DIM))

    # Audio
    if audio_path:
        print(f'  Audio ← {audio_path}')
        audio_seq = extract_audio(audio_path)
    else:
        print('  Audio → using zeros (no audio provided)')
        audio_seq = np.zeros((10, AUDIO_DIM))

    # Text
    if text:
        print(f'  Text  ← "{text[:60]}{"..." if len(text)>60 else ""}"')
        text_feat = extract_text(text)
    else:
        print('  Text  → using zeros (no text provided)')
        text_feat = np.zeros(796)

    # ── Align temporal dimensions ─────────────────────────────────────────────
    T = min(len(face_seq), len(audio_seq))
    face_seq  = face_seq[:T]
    audio_seq = audio_seq[:T]
    print(f'\n  Temporal frames: {T}')

    # ── Compute incongruence embeddings per frame ─────────────────────────────
    inc_seq = []
    with torch.no_grad():
        for t in range(T):
            fp = np.zeros(FACE_DIM)
            fp[:min(len(face_seq[t]), FACE_DIM)] = face_seq[t][:min(len(face_seq[t]), FACE_DIM)]

            ap = np.zeros(AUDIO_DIM)
            ap[:min(len(audio_seq[t]), AUDIO_DIM)] = audio_seq[t][:min(len(audio_seq[t]), AUDIO_DIM)]

            tp = np.zeros(796)
            tp[:min(len(text_feat), 796)] = text_feat[:min(len(text_feat), 796)]

            fv = torch.FloatTensor(fp).unsqueeze(0)
            av = torch.FloatTensor(ap).unsqueeze(0)
            tv = torch.FloatTensor(tp).unsqueeze(0)

            inc_seq.append(inc(fv, av, tv).numpy()[0])

    inc_seq = np.array(inc_seq)

    # ── Compute ECI ───────────────────────────────────────────────────────────
    if len(inc_seq) >= 20:
        seq_t = torch.FloatTensor(inc_seq[:20]).unsqueeze(0)
        with torch.no_grad():
            eci = temp(seq_t).numpy()[0][0]
    else:
        eci = float(np.mean(np.linalg.norm(inc_seq, axis=1)))

    # ── Additional metrics ────────────────────────────────────────────────────
    temporal_variance      = float(np.var(inc_seq))
    incongruence_magnitude = float(np.mean(np.linalg.norm(inc_seq, axis=1)))

    # ── Results ───────────────────────────────────────────────────────────────
    print('\n' + '='*45)
    print('EXPRESSIVE CONTROL INDEX (ECI) RESULTS')
    print('='*45)
    print(f'  ECI Score:               {eci:.4f}')
    print(f'  Temporal Variance:       {temporal_variance:.4f}')
    print(f'  Incongruence Magnitude:  {incongruence_magnitude:.4f}')
    print('='*45)

    if eci >= 0.8:
        label = 'HIGH expressive control — signals are congruent and stable'
    elif eci >= 0.5:
        label = 'MODERATE expressive control — some cross-modal inconsistency'
    else:
        label = 'LOW expressive control — high cross-modal incongruence detected'

    print(f'\n  Interpretation: {label}')
    print(f'\n  Reference ranges (from training):')
    print(f'    IEMOCAP mean ECI: 0.9228  (controlled lab speech)')
    print(f'    MOSEI mean ECI:   0.6774  (naturalistic YouTube speech)')

    return {
        'eci':                    round(float(eci), 4),
        'temporal_variance':      round(temporal_variance, 4),
        'incongruence_magnitude': round(incongruence_magnitude, 4),
        'interpretation':         label
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ECI score from multimodal input')
    parser.add_argument('--video', type=str, default=None, help='Path to video file (.mp4/.avi)')
    parser.add_argument('--audio', type=str, default=None, help='Path to audio file (.wav/.mp3)')
    parser.add_argument('--text',  type=str, default=None, help='Transcript text (in quotes)')
    args = parser.parse_args()

    if not any([args.video, args.audio, args.text]):
        print('ERROR: Provide at least one of --video, --audio, --text')
        parser.print_help()
        sys.exit(1)

    run_inference(
        video_path=args.video,
        audio_path=args.audio,
        text=args.text
    )
