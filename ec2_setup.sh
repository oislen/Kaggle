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
# note: right click to copy paste into putty

# STEP 1: Cinfugure EC2 Instance
# check available memory and cpu capacity
free -h
df -h
lscpu
# reset premission for the instance
ls -larth /.
sudo chmod -R 777 /opt /dev /run /sys/fs/cgroup
ls -larth /.
# install git
sudo yum update -y
sudo yum install git-all -y
# git clone aws setup repo
git clone https://github.com/oislen/AWS.git
# configure vim
sudo yum install vim
git clone https://github.com/gmarik/Vundle.vim.git ~/.vim/bundle/Vundle.vim
cp ~/AWS/.vimrc ~/.
# Note need to install plugins
#vim AWS/ec2_setup.sh
#:PluginInstall
#:q
#sudo yum install curl vim exuberant-ctags git ack-grep
#sudo pip install pep8 flake8 pyflakes isort yapf
#touch ~/.vimrc

# STEP 2: Configure GUI
# install tigervnc
cat /etc/os-release
sudo amazon-linux-extras install mate-desktop1.x
sudo bash -c 'echo PREFERRED=/usr/bin/mate-session > /etc/sysconfig/desktop'
echo "/usr/bin/mate-session" > ~/.Xclients && chmod +x ~/.Xclients
sudo yum install tigervnc-server
vncpasswd
#n
vncserver :1
sudo cp /lib/systemd/system/vncserver@.service /etc/systemd/system/vncserver@.service
sudo sed -i 's/<USER>/ec2-user/' /etc/systemd/system/vncserver@.service
sudo systemctl daemon-reload
sudo systemctl enable vncserver@:1
sudo systemctl start vncserver@:1
# switch over to gui

# STEP 2: Install Conda Environment
# download and install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
yes
<ENTER>  
/dev/anaconda3
yes
# reset terminal
exit

# STEP 3: Create Conda Environement
# configure conda
# auto create aws environment
#export PATH=/dev/anaconda3/bin:$PATH
#conda init bash
conda config --set auto_activate_base false
conda deactivate
conda create --yes --name aws
conda activate aws
conda install --yes pandas
conda install --yes scipy
conda install --yes scikit-learn
conda install --yes statsmodels
conda install --yes seaborn
conda install --yes spyder
conda install --yes notebook
conda install --yes pyarrow
pip install pygam
conda list
#conda env export > aws.yml
#conda env create -f aws.yml


# STEP 4: Create Kaggle Repo
# create kaggle scripts & data
# upload raw data to ec2
cd /run
sudo git clone https://github.com/oislen/Kaggle.git 
sudo mkdir -p /run/Kaggle/Predict_Future_Sales/data/raw
sudo mkdir -p /run/Kaggle/Predict_Future_Sales/data/clean
sudo mkdir -p /run/Kaggle/Predict_Future_Sales/data/base
sudo mkdir -p /run/Kaggle/Predict_Future_Sales/data/model
sudo mkdir -p /run/Kaggle/Predict_Future_Sales/data/pred
sudo mkdir -p /run/Kaggle/Predict_Future_Sales/data/ref
sudo mkdir -p /run/Kaggle/Predict_Future_Sales/report
cd /
sudo chmod 777 -R /run/Kaggle

# STEP 5: UPLOAD Data
# via winscp
#sudo cp -rf /home/ec2-user/raw/ /opt/Kaggle/Predict_Future_Sales/data/

