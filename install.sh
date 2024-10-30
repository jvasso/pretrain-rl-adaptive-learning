conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y pyg -c pyg
conda install -y -c conda-forge gymnasium
conda install -y -c conda-forge tianshou
conda install -y -c conda-forge pyyaml
conda install -y -c conda-forge matplotlib
conda install -y -c anaconda seaborn
conda install -y conda-forge::wandb
pip install git+https://github.com/jvasso/experiments_utils.git