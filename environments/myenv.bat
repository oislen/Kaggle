:: list all available conda environments
call conda env list

:: create and activate new environment
call conda env remove --name myenv
call conda create --name myenv python=3.6 --yes
call conda activate myenv
call conda list

:: install all relevant python libraries
call conda install -c conda-forge numpy --yes
call conda install -c conda-forge pandas --yes
call conda install -c conda-forge scipy --yes
call conda install -c conda-forge statsmodels --yes
call conda install -c conda-forge scikit-learn --yes
call conda install -c conda-forge matplotlib --yes
call conda install -c conda-forge seaborn --yes
::call conda install -c conda-forge keras --yes
call conda install -c conda-forge spyder --yes
call conda install -c conda-forge notebook --yes
call conda install -c conda-forge pyarrow --yes
call conda install -c conda-forge pygam --yes
call conda install -c conda-forge mysqlalchemy --yes
call conda install -c conda-forge scrapy --yes

:: pip install additional libraries
call pip install kaggle

:: list all installed libraries
call conda list

:: export to yml file
:: conda env export > aws.yml
:: conda env create -f aws.yml

::call conda deactivate