# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:41:39 2021

@author: oislen
"""

# load relevant libraries
import cons
from download_comp_data import download_comp_data
     
# download competition data
download_comp_data(comp_name = cons.comp_name,
                   data_dir = cons.data_dir,
                   download_data = cons.download_data, 
                   unzip_data = cons.unzip_data, 
                   del_zip = cons.del_zip
                   )
