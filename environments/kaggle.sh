# list all available conda environments
conda env list

# create and activate new environment
conda env remove --name kaggle
conda env list
conda create --name kaggle python=3.6 --yes
conda activate kaggle
conda list

# pip install additional libraries
pip install --upgrade pip
pip install kaggle

# install all relevant python libraries
conda install -c conda-forge numpy --yes
conda install -c conda-forge scipy --yes
conda install -c conda-forge pandas --yes
#conda install -c conda-forge spyder --yes
#conda install -c conda-forge notebook --yes
conda install -c conda-forge pyarrow --yes

# visualisation libraries
conda install -c conda-forge matplotlib --yes
conda install -c conda-forge seaborn --yes

# nlp libraries
conda install -c conda-forge langdetect --yes
conda install -c conda-forge pyspellchecker --yes
conda install -c conda-forge nltk --yes
conda install -c conda-forge gensim --yes
conda install -c conda-forge sentencepiece --yes
conda install -c conda-forge spacy --yes
python -m spacy download en_core_web_sm

# ml libraries
conda install -c conda-forge statsmodels --yes
conda install -c conda-forge scikit-learn --yes
conda install -c conda-forge pygam --yes
conda install -c conda-forge keras --yes
# conda install -c conda-forge tensorflow --yes
# conda install -c conda-forge tensorflow-hub --yes

# list all installed libraries
conda list

# export to yml file
# conda env export > kaggle.yml
# conda env create -f kaggle.yml

#conda deactivate