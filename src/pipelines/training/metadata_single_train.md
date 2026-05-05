single_train_0.py:
- Single subject
- No ICA
- Feature Extraction: STFT

single_train_1.py:
- Single subject
- No ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum

single_train_1.py:
- Single subject
- No ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 256
- 'joint_dim': 128

single_train_3.py:
- Single subject
- No ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 256
- 'joint_dim': 128
- Normalization

single_train_4.py:
- Single subject
- No ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 128
- 'joint_dim': 128
- Normalization
- Overlapping segment

single_train_5.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 128
- 'joint_dim': 128
- Normalization
- Overlapping segment (hop=16, win=32)

single_train_6.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 128
- 'joint_dim': 128
- Normalization
- Overlapping segment (hop=8, win=16)

single_train_7.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 256
- 'decoder_dim': 128
- 'joint_dim': 256
- Normalization
- Overlapping segment (hop=8, win=16)

single_train_8.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 256
- 'decoder_dim': 128
- 'joint_dim': 128
- Normalization
- Overlapping segment (hop=8, win=16)

single_train_9.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 768
- 'decoder_dim': 768
- 'joint_dim': 768
- Normalization
- Overlapping segment (hop=8, win=16)
- Decoder: IndoGPT

single_train_10.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 768
- 'joint_dim': 768
- Normalization
- Overlapping segment (hop=8, win=16)
- Decoder: IndoGPT

single_train_11.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 768
- 'joint_dim': 768
- Normalization
- Overlapping segment (hop=32, win=64)
- Decoder: IndoGPT

single_train_12.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 128
- 'decoder_dim': 768
- 'joint_dim': 768
- Normalization
- Overlapping segment (hop=16, win=32)
- Decoder: IndoGPT

single_train_13.py:
- Single subject
- USE ICA
- Feature Extraction: CEEMDAN + Hilbert Spectrum
- fmin=0.2, fmax=45.0
- 'encoder_dim': 256
- 'decoder_dim': 768
- 'joint_dim': 768
- Normalization
- Overlapping segment (hop=16, win=32)
- Decoder: IndoGPT