# install git
sudo yum install git-all

# git clone aws setup repo
git clone https://github.com/oislen/AWS.git

# configure vim
sudo yum update
sudo yum install vim
git clone https://github.com/gmarik/Vundle.vim.git ~/.vim/bundle/Vundle.vim
cp AWS/.vimrc ~/.
vim AWS/aws_ec2_setup.sh
:PluginInstall
:q
#sudo yum install curl vim exuberant-ctags git ack-grep
#sudo pip install pep8 flake8 pyflakes isort yapf
#touch ~/.vimrc

# download and install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
yes
<ENTER>
yes
#reset terminal

# configure conda
export PATH=~/anaconda3/bin:$PATH
conda init bash
# reset terminal

# create kaggle environment
#conda create --name aws
#conda activate kaggle
#conda install pandas
#conda install scipy
#conda install scikit-learn
#conda install statsmodels
#conda install seaborn
#pip install pygam
#conda env export > aws.yml
conda env create -f aws.yml
# reset terminal

# install tigervnc
sudo yum update -y
sudo yum install -y pixman pixman-devel libXfont
sudo yum -y install tigervnc-server
vncpasswd
sudo service sshd restart
sudo vi /etc/sysconfig/vncservers
sudo service vncserver start
sudo chkconfig vncserver on
#vncserver :1