export HF_HOME=/home/huggingface_hub/
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl --force-reinstall --no-deps
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# Install forked transformers
git clone https://github.com/quic-meetkuma/transformers.git
cd transformers
git checkout qaic_support_transformer_v5.1-release
pip install -e .

cd ..

# Install forked accelerate
git clone https://github.com/quic-meetkuma/accelerate.git
cd accelerate
git checkout v1.12.0-release-shubham-changes-dp-tp
pip install -e .

cd ..

# Install PEFT and SFT related libraries
pip install trl==0.28.0 datasets==4.5.0 peft==0.18.1
