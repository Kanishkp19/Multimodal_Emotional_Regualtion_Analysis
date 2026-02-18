# 🚀 IMPLEMENTATION GUIDE
## Multimodal Emotion Regulation Analysis

Complete step-by-step guide to set up and run the entire project.

---

## 📋 Prerequisites

- **Hardware**: Apple M4 Mac (or any modern CPU)
- **RAM**: 16 GB
- **Storage**: ~50-60 GB free space
- **OS**: macOS
- **Python**: 3.9 or higher

---

## 🔧 STEP 1: Environment Setup

### 1.1 Navigate to Project Directory

```bash
cd /path/to/emotion-regulation-analysis
```

### 1.2 Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Your terminal should now show `(venv)` prefix.

### 1.3 Upgrade pip

```bash
pip install --upgrade pip
```

### 1.4 Install Dependencies

```bash
pip install -r requirements.txt
```

⏱️ **Expected time**: 5-10 minutes

**Note**: If you encounter any errors:
- For mediapipe issues on M4: `pip install mediapipe --no-cache-dir`
- For torch issues: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

---

## 🔑 STEP 2: Setup Kaggle API (for IEMOCAP)

### 2.1 Get Kaggle API Credentials

1. Go to: https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`

### 2.2 Install Kaggle Credentials

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2.3 Verify Setup

```bash
kaggle --version
```

You should see the Kaggle version number.

---

## 📦 STEP 3: Download Datasets

### 3.1 Download IEMOCAP

```bash
python preprocessing/download_iemocap.py
```

⏱️ **Expected time**: 15-30 minutes
💾 **Size**: ~12 GB

**Expected output**:
```
✓ Kaggle credentials found
📁 Download location: ./data/iemocap
⏳ Starting download...
...
✅ IEMOCAP download successful!
```

**Troubleshooting**:
- If download fails: Check internet connection
- If "dataset not found": Go to kaggle.com and accept the dataset terms
- If permission denied: Re-run chmod command

### 3.2 Download CMU-MOSEI

```bash
python preprocessing/download_mosei.py
```

⏱️ **Expected time**: 20-40 minutes
💾 **Size**: ~15-20 GB

**Expected output**:
```
✓ CMU Multimodal SDK installed
⏳ Downloading MOSEI dataset components...
...
✅ CMU-MOSEI setup successful!
```

**Troubleshooting**:
- If SDK installation fails: The script will fall back to manual download
- Large file warning is normal

---

## 🎯 STEP 4: Feature Extraction

### 4.1 Extract IEMOCAP Features

```bash
python preprocessing/extract_features.py --dataset iemocap
```

⏱️ **Expected time**: 30-60 minutes (depends on dataset size)

**What's happening**:
- Processing videos with MediaPipe (face landmarks)
- Extracting audio with Librosa (MFCC + prosody)
- Processing text with RoBERTa (emotion features)

**Expected output**:
```
Initializing encoders...
✓ Face Encoder initialized (MediaPipe)
✓ Audio Encoder initialized (MFCC + Prosody)
✓ Text Encoder initialized (RoBERTa + GoEmotions)

Processing Session1...
Processing Session2...
...

✓ Extracted XXXX samples
✓ Features saved to data/processed/iemocap_features.pkl
```

### 4.2 Extract MOSEI Features

```bash
python preprocessing/extract_features.py --dataset mosei
```

⏱️ **Expected time**: 20-40 minutes

**Expected output**:
```
Loading MOSEI data using CMU SDK...
Processing MOSEI samples...
✓ Extracted XXXX samples
✓ Features saved to data/processed/mosei_features.pkl
```

---

## 📊 STEP 5: Create Processed Datasets

```bash
python preprocessing/create_dataset.py --dataset both
```

⏱️ **Expected time**: 2-5 minutes

**What's happening**:
- Aligning temporal dimensions across modalities
- Aggregating features to utterance level
- Creating train/val splits for IEMOCAP
- Preparing test set for MOSEI

**Expected output**:
```
Creating IEMOCAP Dataset
✓ Aggregated features:
  Face: (XXXX, 944)
  Audio: (XXXX, 42)
  Text: (XXXX, 796)

✓ Created train/val splits:
  Train: XXXX samples -> data/processed/iemocap_train.pkl
  Val: XXXX samples -> data/processed/iemocap_val.pkl
  Temporal: XXXX sequences -> data/processed/iemocap_temporal.pkl

Creating MOSEI Dataset
✓ Created test set:
  Test: XXXX samples -> data/processed/mosei_test.pkl
```

---

## 🎓 STEP 6: Train Incongruence Encoder

```bash
python training/train_incongruence.py \
    --epochs 15 \
    --batch_size 32 \
    --lr 0.001
```

⏱️ **Expected time**: 20-40 minutes on M4 Mac

**What's happening**:
- Self-supervised contrastive learning
- Learning cross-modal incongruence patterns
- Training via aligned vs misaligned samples

**Expected output**:
```
Training Incongruence Encoder
✓ Train samples: XXXX
✓ Val samples: XXXX
✓ Model parameters: XXX,XXX

Starting training...

Epoch 1/15
train_loss: 0.XXXX | val_loss: 0.XXXX | learning_rate: 0.001000
  ✓ Saved best model (val_loss: 0.XXXX)

Epoch 2/15
...

Training complete!
Best validation loss: 0.XXXX
Model saved to: ./checkpoints
```

**Monitoring**:
```bash
# In a separate terminal:
tensorboard --logdir logs/incongruence_encoder
```
Open browser to: http://localhost:6006

---

## ⏰ STEP 7: Train Temporal Model

```bash
python training/train_temporal.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --sequence_length 20
```

⏱️ **Expected time**: 30-50 minutes on M4 Mac

**What's happening**:
- Loading pre-trained incongruence encoder (frozen)
- Extracting incongruence features from sequences
- Training Bi-LSTM for temporal aggregation
- Learning to compute ECI from temporal patterns

**Expected output**:
```
Training Temporal Model
Loading incongruence encoder...
✓ Incongruence encoder loaded

Extracting incongruence features...
100%|████████████| XXXX/XXXX
✓ Created XXXX temporal samples
✓ Train samples: XXXX
✓ Val samples: XXXX
✓ Model parameters: XXX,XXX

Starting training...

Epoch 1/20
train_loss: 0.XXXX | val_loss: 0.XXXX | learning_rate: 0.001000
  ✓ Saved best model (val_loss: 0.XXXX)

...

Training complete!
Best validation loss: 0.XXXX
Model saved to: ./checkpoints
```

**Monitoring**:
```bash
tensorboard --logdir logs/temporal_model
```

---

## 📈 STEP 8: Evaluate Models

```bash
python evaluate.py --dataset both
```

⏱️ **Expected time**: 10-20 minutes

**What's happening**:
- Loading both trained models
- Computing ECI scores for IEMOCAP
- Computing ECI scores for MOSEI
- Comparing distributions across datasets

**Expected output**:
```
Evaluating Emotion Regulation Analysis

Loading trained models...
✓ Models loaded successfully

Processing iemocap_temporal.pkl...
Computing ECI: 100%|████████████| XXXX/XXXX
✓ Computed XXXX ECI scores

IEMOCAP Results
  mean: 0.XXXX
  std: 0.XXXX
  median: 0.XXXX
  ...

✓ Results saved to results/eci_scores_iemocap.csv

Processing mosei_test.pkl...
...

Cross-Dataset Comparison
  ks_statistic: 0.XXXX
  mannwhitney_pvalue: 0.XXXX
  cohens_d: X.XXXX
  ...

Evaluation Complete!
Results saved to: ./results
```

**Output files**:
- `results/eci_scores_iemocap.csv` - ECI scores for each IEMOCAP sample
- `results/eci_scores_mosei.csv` - ECI scores for each MOSEI sample
- `results/cross_dataset_comparison.json` - Statistical comparison

---

## 📊 STEP 9: Run Analysis (Optional)

### 9.1 Start Jupyter

```bash
jupyter notebook
```

### 9.2 Open Notebooks

1. **Visualization Notebook**: `analysis/visualization.ipynb`
   - ECI distribution plots
   - Temporal dynamics visualization
   - Cross-modal incongruence heatmaps

2. **Ablation Study Notebook**: `analysis/ablation_study.ipynb`
   - Modality ablation (remove face/audio/text)
   - Temporal ablation (with/without LSTM)
   - Sequence length analysis

---

## 🎯 Quick Start (Minimum Viable Run)

If you want to test the pipeline quickly with a subset:

```bash
# 1. Setup environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download small dataset (IEMOCAP only)
python preprocessing/download_iemocap.py

# 3. Extract features (may take time)
python preprocessing/extract_features.py --dataset iemocap

# 4. Create dataset
python preprocessing/create_dataset.py --dataset iemocap

# 5. Train incongruence encoder (quick test: 5 epochs)
python training/train_incongruence.py --epochs 5

# 6. Train temporal model (quick test: 10 epochs)
python training/train_temporal.py --epochs 10

# 7. Evaluate
python evaluate.py --dataset iemocap
```

---

## 🐛 Common Issues & Solutions

### Issue: "Kaggle API not working"
**Solution**: 
```bash
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list  # Test if working
```

### Issue: "Out of memory during feature extraction"
**Solution**: Process in smaller batches by modifying extract_features.py:
```python
# Line ~80, change to:
for video_file in tqdm(video_files[:100], desc=f"  {session_dir.name}"):
```

### Issue: "MediaPipe not detecting faces"
**Solution**: Check video quality, lighting. Skip problematic videos (already handled in code).

### Issue: "Training is very slow"
**Solution**: 
- Reduce batch_size: `--batch_size 16`
- Reduce epochs for testing: `--epochs 5`
- Monitor with `top` command to check CPU usage

### Issue: "Model not learning (loss not decreasing)"
**Solution**:
- Check if features were extracted correctly
- Verify dataset has enough samples (>100)
- Try different learning rate: `--lr 0.0001`

---

## 📁 Expected Directory Structure After Completion

```
emotion-regulation-analysis/
├── data/
│   ├── iemocap/          # Downloaded IEMOCAP data
│   ├── mosei/            # Downloaded MOSEI data
│   └── processed/        # Extracted features
│       ├── iemocap_features.pkl
│       ├── iemocap_train.pkl
│       ├── iemocap_val.pkl
│       ├── iemocap_temporal.pkl
│       └── mosei_test.pkl
├── checkpoints/          # Trained models
│   ├── incongruence_encoder_best.pth
│   └── temporal_model_best.pth
├── logs/                 # TensorBoard logs
│   ├── incongruence_encoder/
│   └── temporal_model/
├── results/              # Evaluation results
│   ├── eci_scores_iemocap.csv
│   ├── eci_scores_mosei.csv
│   └── cross_dataset_comparison.json
└── [source files...]
```

---

## ⏱️ Total Time Estimate

| Step | Time |
|------|------|
| Environment Setup | 10 min |
| Dataset Download | 45-70 min |
| Feature Extraction | 50-100 min |
| Dataset Creation | 5 min |
| Train Incongruence | 20-40 min |
| Train Temporal | 30-50 min |
| Evaluation | 10-20 min |
| **TOTAL** | **~3-5 hours** |

---

## 💡 Pro Tips

1. **Run overnight**: Dataset download + feature extraction can be left running overnight
2. **Use tmux/screen**: For long-running processes on remote machines
3. **Monitor with TensorBoard**: Always check training curves
4. **Save intermediate results**: The pipeline saves at each step so you can resume
5. **Test with subset first**: Use `[:100]` slicing to test pipeline quickly

---

## 📞 Support

If you encounter issues not covered here:
1. Check error messages carefully
2. Verify file paths and permissions
3. Ensure all previous steps completed successfully
4. Check `logs/` directory for detailed logs

---

## ✅ Success Checklist

- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Kaggle API configured
- [ ] IEMOCAP downloaded
- [ ] MOSEI downloaded
- [ ] Features extracted for IEMOCAP
- [ ] Features extracted for MOSEI
- [ ] Processed datasets created
- [ ] Incongruence encoder trained
- [ ] Temporal model trained
- [ ] Evaluation completed
- [ ] Results files generated
- [ ] Analysis notebooks reviewed

---

**You're all set! The project is now fully operational.**