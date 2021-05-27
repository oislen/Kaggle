# list all available conda environments
conda env list

# create and activate new environment
conda env remove --name kaggle
conda create --name kaggle python=3.6 --yes
conda activate kaggle
conda list

# install all relevant python libraries
conda install -c conda-forge numpy --yes
conda install -c conda-forge pandas --yes
conda install -c conda-forge scipy --yes
conda install -c conda-forge statsmodels --yes
conda install -c conda-forge scikit-learn --yes
conda install -c conda-forge matplotlib --yes
conda install -c conda-forge seaborn --yes
#conda install -c conda-forge keras --yes
conda install -c conda-forge spyder --yes
conda install -c conda-forge notebook --yes
conda install -c conda-forge pyarrow --yes
conda install -c conda-forge pygam --yes
#conda install -c conda-forge mysqlalchemy --yes
#conda install -c conda-forge scrapy --yes
conda install -c conda-forge spacy --yes
python -m spacy download en_core_web_sm

# pip install additional libraries
pip install kaggle
pip install langdetect
pip install pyspellchecker
pip install nltk
pip install --upgrade gensim

# list all installed libraries
conda list

# export to yml file
# conda env export > aws.yml
# conda env create -f aws.yml

#conda deactivate