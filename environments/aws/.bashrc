# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
alias ll='ls -larth'

function colcnt(){
        fpath=$1
        sep=$2
        awk -F$sep '{print NF; exit}' $fpath
}

function rowcnt(){
        fpath=$1
        wc -l $fpath | awk {'print $fpath}'
}

function ram(){
        free -m | awk 'FNR == 2 {print $3/($3+$4)*100}'
}
