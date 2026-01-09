export HF_HOME=/home/huggingface_hub/
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl --force-reinstall --no-deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install trl datasets peft 
pip install git+https://github.com/quic-meetkuma/transformers.git@9cd1f690c95cb526600dd0d4ab32bf7d4a58d720#egg=transformers -e .
pip install git+https://github.com/quic-meetkuma/accelerate.git@4ebcbddc01be1b7441fc1ee9ba9b9fd474fdcb14#egg=accelerate -e .
