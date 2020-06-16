# AWS Setup
# Big Data Workloads
# Amazon Linux 2 with .Net Core, PowerShell, Mono, and MATE Desktop Environment
# c3.4xlarge (16 vCPUs, 30GiB, 2 x 160 SSD)
# eu-west-1c
# ec2-user
# useful links
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connection-prereqs.html#connection-prereqs-get-info-about-instance
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html
# https://docs.aws.amazon.com/transfer/latest/userguide/getting-started-use-the-service.html
# https://aws.amazon.com/premiumsupport/knowledge-center/ec2-linux-2-install-gui/
# note: winscp sftp settings: sudo /usr/lib/openssh/sftp-server

# check available memory
free
df

# reset premission for the instance
sudo chmod -R 777 /.

# install git
sudo yum update
sudo yum install git-all

# git clone aws setup repo
git clone https://github.com/oislen/AWS.git

# configure vim
sudo yum install vim
git clone https://github.com/gmarik/Vundle.vim.git ~/.vim/bundle/Vundle.vim
cp ~/AWS/.vimrc ~/.
vim AWS/ec2_setup.sh
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
conda config --set auto_activate_base false
# reset terminal

# auto create aws environment
conda deactivate
conda create --yes --name aws
conda activate aws
conda install --yes pandas
conda install --yes scipy
conda install --yes scikit-learn
conda install --yes statsmodels
conda install --yes seaborn
pip install pygam
#conda env export > aws.yml
#conda env create -f aws.yml
# reset terminal

# create kaggle scripts & data
# upload raw data to ec2
sudo git clone https://github.com/oislen/Kaggle.git
sudo mkdir /opt/Kaggle/Predict_Future_Sales/data
sudo mkdir /opt/Kaggle/Predict_Future_Sales/report
sudo cp -rf /home/ec2-user/raw/ /opt/Kaggle/Predict_Future_Sales/data/

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
