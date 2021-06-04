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
# calculate percentage of used memory
free -m | awk 'FNR == 2 {print $3/($3+$4)*100}'
# reset premission for the instance
ls -larth /.
sudo chmod -R 777 /opt /dev /run /sys/fs/cgroup
sudo chmod 775 /var/run/screen
ls -larth /.
# install git
cd ~
sudo yum update -y
sudo yum install git-all -y
# git clone the kaggle repo
git clone https://github.com/oislen/Kaggle.git /dev/Kaggle
# configure vim
sudo yum install vim
git clone https://github.com/gmarik/Vundle.vim.git ~/.vim/bundle/Vundle.vim
cp /dev/Kaggle/environments/aws/.vimrc ~/.
vim -c ':PluginInstall' -c ':q' -c ':q'
sudo yum install curl vim exuberant-ctags git ack-grep
sudo easy_install pip
sudo pip install pep8 flake8 pyflakes isort yapf
#touch ~/.vimrc
# configure .bashrc file
cp /dev/Kaggle/environments/aws/.bashrc ~/.
source .bashrc
# install htop
sudo yum install htop -y
# update overcommit memory setting
cat /proc/sys/vm/overcommit_memory
echo 1 | sudo tee /proc/sys/vm/overcommit_memory


# STEP 2: Configure GUI (optional)
if [ 0 -eq 1 ]
then
	# install tigervnc
	#screen
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
fi

# STEP 3: Install Anaconda
# download and install anaconda
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2021.04-Linux-x86_64.sh
bash Anaconda3-2021.04-Linux-x86_64.sh
# required manual inputs for anaconda installation:
# yes
# <ENTER>  
# /dev/anaconda3
# yes
# next reset terminal
exit

# STEP 4: Create Conda Environement
# configure conda
# auto create aws environment
#export PATH=/dev/anaconda3/bin:$PATH
conda init bash
source /dev/anaconda3/etc/profile.d/conda.sh
conda config --set auto_activate_base false
# install encoding editing for windows scripts
sudo yum install dos2unix -y
dos2unix /dev/Kaggle/environments/kaggle.sh
bash /dev/Kaggle/environments/kaggle.sh

# STEP 5: Create Kaggle Repo
# create kaggle scripts & data
sudo chmod 777 -R /dev/Kaggle
cd /dev/Kaggle/utilities/comp
python gen_kaggle_dirs.py

# STEP 6: UPLOAD Data
# via winscp
# /home/ec2-user/.kaggle/kaggle.json (possibly automate with scp / sftp)
# chmod 600 /home/ec2-user/.kaggle/kaggle.json

