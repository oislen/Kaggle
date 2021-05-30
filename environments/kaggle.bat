:: list all available conda environments
call conda env list

:: create and activate new environment
call conda env remove --name kaggle
call conda env list
call conda create --name kaggle python=3.6 --yes
call conda activate kaggle
call conda list

:: pip install additional libraries
call pip install --upgrade pip
call pip install kaggle

:: install all relevant python libraries
call conda install -c conda-forge numpy --yes
call conda install -c conda-forge scipy --yes
call conda install -c conda-forge pandas --yes
call conda install -c conda-forge spyder --yes
call conda install -c conda-forge notebook --yes
call conda install -c conda-forge pyarrow --yes

:: visualisation libraries
call conda install -c conda-forge matplotlib --yes
call conda install -c conda-forge seaborn --yes

:: nlp libraries
call conda install -c conda-forge langdetect --yes
call conda install -c conda-forge pyspellchecker --yes
call conda install -c conda-forge nltk --yes
call conda install -c conda-forge gensim --yes
call conda install -c conda-forge sentencepiece --yes
call conda install -c conda-forge spacy --yes
call python -m spacy download en_core_web_sm

:: ml libraries
call conda install -c conda-forge statsmodels --yes
call conda install -c conda-forge scikit-learn --yes
call conda install -c conda-forge pygam --yes
call conda install -c conda-forge keras --yes
:: call conda install -c conda-forge tensorflow --yes
:: call conda install -c conda-forge tensorflow-hub --yes

:: list all installed libraries
call conda list

:: export to yml file
:: conda env export > kaggle.yml
:: conda env create -f kaggle.yml

::call conda deactivate