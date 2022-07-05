from setuptools import find_packages, setup

setup(
    name='connectome',
    packages=find_packages(),
    version='0.1.0',
    description='Connectome Innolab',
    author='Jana Gauß, Jonas Klingele, Kai Becker, Katharina J. Brenner, Leo Schaabner',
    author_email='',
    license='',
    install_requires=['h5py==3.6.0', 'mat73', 'matplotlib==3.5.1', 
                      'numpy==1.22.0', 'pandas==1.3.3', 'scikit_learn==1.0.1', 
                      'seaborn==0.11.2', 'xgboost==1.5.1', 'lightgbm==3.3.1', 
                      'bctpy==0.5.0', 'keras==2.8.0', 'IPython==7.31.1', 
                      'interpret==0.2.7', 'tensorflow==2.8.1', 'tf_keras_vis==0.8.1',
                      'shap==0.40.0', 'openpyxl==3.0.9', 'protobuf==3.20.1']
)
