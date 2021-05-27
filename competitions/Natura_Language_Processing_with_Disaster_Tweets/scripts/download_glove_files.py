# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:21:44 2021

@author: oislen
"""

# import relevant libraries
import cons 
from download_url_files import download_url_files

# download glove files
download_url_files(url = cons.glove_twitter_url, 
                   out_fpath = cons.glove_twitter_fpath
                   )