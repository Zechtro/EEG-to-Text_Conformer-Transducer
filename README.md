# Final Project: Indonesian Imagined Speech Recognition from EEG Signals Using Conformer-Transducer


## Setup Virtual Environment


### Windows
```bash
python -m venv venv_conf
venv_conf\Scripts\activate
pip install -r requirements.txt
```

### MacOS
```bash
python3 -m venv venv_conf
source venv_conf/bin/activate
pip install -r requirements.txt
```

Adjust with your Cuda version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```