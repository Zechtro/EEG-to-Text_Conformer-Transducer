all_train_0:
- All subjects
- No ICA
- Feature Extraction: STFT

all_train_1.py:
- All subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 128
- 'joint_dim': 128
- Normalization
- Overlapping segment (hop=8, win=16)