# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:15:38 2021

@author: oislen
"""

# impoert relevant libraries
import urllib
import zipfile
import os

def download_url_files(url, 
                       out_fpath,
                       download = True,
                       unzip = True,
                       remove = True,
                       print_statements = True
                       ):
    
    """
    
    Download Reference Files Documentation
    
    Overview
    
    This function downloads reference data from a specified url and saves it to the designated out file path.
    
    Defaults
    
    download_ref_files(url, 
                       out_fpath,
                       print_statements = True
                       )
    
    Parameters
    
    url - String, the url to download the reference data
    out_fpath - String, the file path to save the downloaded data
    
    Returns
    
    0 for successfull execution
    
    Example
    
    download_ref_files(url = 'http://airo.maynoothuniversity.ie/files/dDATASTORE/education/csv/post_primary_schools_2013_2014.csv', 
                       out_fpath = 'C:\\Users\\User\\Documents\\GitHub\\IrishPropPricesref_locations\\datapost_primary_schools_2013_2014.csv'
                       )
    
    """
    
    # redefine print function
    def print(x, print_statements = print_statements):
        import builtins as __builtin__
        if print_statements == True:
            __builtin__.print(x)
    
    print('checking programme inputs ...')
    
    # check data type of string based parameters
    if any([type(x) != str for x in [url, out_fpath]]):
        raise ValueError('Input Error: the parameters url, out_fpath must be string data types.')
    
    # if downloading data
    if download:
        
        print('retrieveing url request ...')
        
        # call the url and retrive the file
        urllib.request.urlretrieve(url, out_fpath)
    
    # if unzipping file
    if unzip:
        
        print('Unzipping file ...')
        
        # unzip download
        with zipfile.ZipFile(out_fpath, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(out_fpath))
       
    # if remove file
    if remove:
        
        print('Delete file ...')
        
        # delete zip file
        os.remove(path = out_fpath)
        
    return 0