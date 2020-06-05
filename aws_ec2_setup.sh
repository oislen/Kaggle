# download and install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh

# configure conda
export PATH=~/anaconda3/bin:$PATH
conda init bash

# create kaggle environment
conda create --name kaggle
conda activate kaggle
conda install pandas
conda install scipy
conda install scikit-learn
conda install statsmodels
conda install seaborn
pip install pygam

# configure vim
sudo yum update
sudo yum install vim
#sudo yum install curl vim exuberant-ctags git ack-grep
#sudo pip install pep8 flake8 pyflakes isort yapf
git clone https://github.com/gmarik/Vundle.vim.git ~/.vim/bundle/Vundle.vim
touch ~/.vimrc
