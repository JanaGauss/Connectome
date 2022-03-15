<div id="top"></div>

# Connectome

Created as part of an university project "Innovationlab Big Data Science" at LMU Munich.


<!-- ABOUT THE PROJECT -->
## About The Project

### Abstract

Alzheimer's disease (AD) is a progressive neurologic disorder that causes the brain to shrink (atrophy) and brain cells to die - resulting in reductions of synaptical relations between brain areas (connectivity loss) [1][2]. With over 10 million people estimated to have dementia in Europe with costs ranging up to 175000 US$ per patient [3][4], research in Alzheimer disease will play an distinct role in modern healthcare systems. The aim of this project was therefore two-fold: (i) Prediction  and (ii) Explanation of Alzheimer Diagnosis based on connectivity matrices utlizing the Brainnetome Atlas [5]. The results of the presented endeavours include a pipeline for processing connectivity matrices in order to predict and explain a patients AD status. This pipeline enables the user to automate the training, evaluation and interpretation for several models as well as for several dataset options, e.g. aggregated connectivity matrices, connectivity matrices and graph metrics applied to human brain connectivity data. The applicable models in the pipeline include Elastic Net, Random Forest, Gradient Boosting as well as 2D Convoutional Neural Networks. For Evaluation of the results, the Accuracy, AUC, Precision, REcall and F1 values were compared with the following effects of the Elastic Net (Conn data) as an example: (i) Accuracy: 0.8 (ii) AUC: 0.86 (iii) Precision: 0.88 (iv) F1: 0.82. These results suggest, that the models perform well on the preprocessed connectivity matrices. In a last step, the following brain subregions were identified from the connectivity matrices for their key importance for AD: Caudal Temporal Thalamus, TE1.0 and TE1.2, Dorsal Agranular Insular, Caudal Hippocampus, Rostral Area 7, Posterior Parietal Thalamus, Posterior Parahippocampal Gyrus.





[1] Mayoclinic.org (N.A.). Alzheimer's disease. Retrieved from: https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447#:~:text=Alzheimer's%20disease%20is%20a%20progressive,person's%20ability%20to%20function%20independently. (01.03.22)
[2] Smith, M.A., 1998. Alzheimer disease. International review of neurobiology, 42, pp.1-54.
[3] Alzheimer Europe.org (2019). Dementia in Europe Yearbook 2019. Retrieved from: https://www.alzheimer-europe.org/sites/default/files/alzheimer_europe_dementia_in_europe_yearbook_2019.pdf (20.02.22)
[4] Alzheimer Association´s  (2020). Costs of Alzheimer's to Medicare and Medicaid. Retrieved from: https://act.alz.org/site/DocServer/2012_Costs_Fact_Sheet_version_2.pdf?docID=7161 (22.02.22)
[5] Fan, L., Li, H., Zhuo, J., Zhang, Y., Wang, J., Chen, L., Yang, Z., Chu, C., Xie, S., Laird, A.R. and Fox, P.T., 2016. The human brainnetome atlas: a new brain atlas based on connectional architecture. Cerebral cortex, 26(8), pp.3508-3526.


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow the following steps to start the pipeline locally. 

### Prerequisites

* Install python 3.8.x or above 
* From either:
* https://www.python.org/downloads/ 
* or: https://www.anaconda.com/

**recommended**: Create a virtual environment before installing the package. If you choose not to, you can skip to the Installation
1. Open the command prompt
2. Navigate to the project directory or create a new one:
```
mkdir path/to/python/project
```
3. Create a virtual environment within the project folder
```
python3 -m venv connectome
```
4. Activate virtual environment
```
source connectome/bin/activate
```
5. Go to installation


### Installation

* Windows/Linux
  ```sh
  git clone https://github.com/JanaGauss/Connectome.git
  cd Connectome
  pip install .
  ```



<p align="right">(<a href="#top">back to top</a>)</p>




<!-- USAGE -->
## Usage

This is the beginning to our Connectome Pipeline. 

![image](https://user-images.githubusercontent.com/60140124/158218222-08377392-b718-4c03-8f85-7e911f67f323.png)

For further details check out the demo video or our full documentation here: https://likai97.github.io/Conncetome-Documentation/

### Demo Video

The following video displays the full pipeline from preprocessing to model outputs:

https://user-images.githubusercontent.com/51715552/158160781-d1e6c7bc-de14-49b9-98ce-d7f9198411d1.mp4


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## Authors

Authors:

* Leo Schaabner
* Kai Becker
* Jana Gauß
* Katharina Brenner
* Jonas Klingele

<p align="right">(<a href="#top">back to top</a>)</p>
