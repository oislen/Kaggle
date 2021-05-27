# -*- coding: utf-8 -*-
"""
Created on Thu May 27 08:40:50 2021

@author: oislen
"""

# load in relevant libraries
from spellchecker import SpellChecker

def spell_corrector(text,
                    lang = 'en'
                    ):
    
    """
    
    Spell Corrector Documentation
    
    Function Overview
    
    This function auto corrects the spelling of a given text input and language
    
    Defaults
    
    spell_corrector(text,
                    lang = 'en'
                    )
    
    Parameters
    
    text - String, the text to auto correct the spelling of
    lang - String, the language of the given text to correct
    
    Returns
    
    corrected_text - String, the correct text
    
    Example
    
    spell_corrector(text = 'corect me plse',
                    lang = 'en'
                    )
    """
    
    # create a spell checker instance
    spell = SpellChecker(language = lang, distance = 2)
    
    # create an empty list to holded correct text words
    corrected_text_list = []
    
    # split intput text into seperate words
    text_split = text.split(' ')
    
    # find incorrect words
    misspelled_words = spell.unknown(text_split)
    
    # loop over iput text words
    for word in text_split:
        
        # word is in the misspelled words
        if word in misspelled_words:
            
            # apply the correction
            corrected_word = spell.correction(word)
            
            # append corrected word to list
            corrected_text_list.append(corrected_word)
            
        # otherwise if the word is not misspelled
        else:
            
            # append the word to the list
            corrected_text_list.append(word)
            
    # rejoin the words of the corrected sentenc
    corrected_text = ' '.join(corrected_text_list)
    
    return corrected_text