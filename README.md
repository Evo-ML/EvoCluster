<div align="center">
<img alt="EvoCluster-logo" src="http://evo-ml.com/wp-content/uploads/2021/06/EvoCluster-logo.png" width=40%>
</div>

# An Open-Source Nature-Inspired Optimization Clustering Framework in Python
## Description ##
EvoCluster is an open source and cross-platform framework implemented in Python which includes the most well-known and recent nature-inspired meta heuristic  optimizers  that  are  customized  to  perform  partitional  clustering tasks.  The  goal  of  this  framework  is  to  provide  a  user-friendly  and  customizable implementation of the metaheuristic based clustering algorithms which canbe utilized by experienced and non-experienced users for different applications.The framework can also be used by researchers who can benefit from the implementation of the metaheuristic optimizers for their research studies. EvoClustercan be extended by designing other optimizers, including more objective func-tions, adding other evaluation measures, and using more data sets. The current implementation of the framework includes ten metaheuristic optimizers, thirty datasets, five objective functions, twelve evaluation measures, more than twenty distance measures, and ten different ways for detecting the k value. The  source code of EvoCluster is publicly available at (http://evo-ml.com/evocluster/).

## Versions ##
- <a href='https://pypi.org/project/EvoCluster/'>1.0.5</a>

## Supporting links
- **Web Page**: http://evo-ml.com/evocluster/
- **Paper**: https://link.springer.com/chapter/10.1007/978-3-030-43722-0_2
- **Extended Paper**: https://link.springer.com/article/10.1007/s42979-021-00511-0
- **Introduction video**: https://www.youtube.com/watch?v=3DYIdxILZaw
- **Demo video**: https://www.youtube.com/watch?v=TOIo9WMBWUc
- **Source code**: https://github.com/RaneemQaddoura/EvoCluster/
- **Bug reports**:  https://github.com/RaneemQaddoura/EvoCluster/issues

## Features
- Ten nature-inspired metaheuristic optimizers are implemented (SSA, PSO, GA, BAT, FFA, GWO, WOA, MVO, MFO, and CS).
- Five objective functions (SSE, TWCV, SC, DB, and DI).
- Thirty datasets obtained from Scikit learn, UCI, School of Computing at University of Eastern Finland, ELKI, KEEL, and Naftali Harris Blog
- Twelve evaluation measures (SSE, Purity,	Entropy,	HS,	CS,	VM,	AMI,	ARI,	Fmeasure,	TWCV,	SC,	Accuracy,	DI,	DB,	and Standard Diviation)
- More than twenty distance measures
- Ten different ways for detecting the k value
- The implimentation uses the fast array manipulation using [`NumPy`] (http://www.numpy.org/).
- Matrix support using [`SciPy`'s] (https://www.scipy.org/) package.
- Simple and efficient tools for prediction using [`sklearn`] (https://scikit-learn.org/stable/)
- File data analysis and manipulation tool using [`pandas`] (https://pandas.pydata.org/)
- Plot interactive visualizations using [`matplotlib`] (https://matplotlib.org/)
- More optimizers, objective functions, adatasets, and evaluation measures are comming soon.
 
## Installation
EvoCluster supports Python 3.xx.
 
### With pip
EvoCluster can be installed using pip as follows:

```bash
pip install EvoCluster
```
### With conda
EvoCluster can be installed using conda as follows:

```shell script
conda install EvoCluster
```
### To install from master
```
pip install git+https://github.com/RaneemQaddoura/EvoCluster.git#egg=EvoCluster
```
### Get the source

Clone the Git repository from GitHub
```bash
git clone https://github.com/RaneemQaddoura/EvoCluster.git
```

## Quick tour

To immediately use a EvoCluster. Here is how to quickly use EvoCluster to predict clusters:

```python
from EvoCluster import EvoCluster

optimizer = ["SSA", "PSO", "GA", "GWO"] #Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS".
objective_func = ["SSE", "TWCV"] #Select objective function from the list of available ones:"SSE","TWCV","SC","DB","DI".
dataset_list = ["iris", "aggregation"] #Select data sets from the list of available ones
num_of_runs = 3 #Select number of repetitions for each experiment. 
params = {'PopulationSize': 30, 'Iterations': 50} #Select general parameters for all optimizers (population size, number of iterations)
export_flags = {'Export_avg': True, 'Export_details': True, 'Export_details_labels': True,
                'Export_convergence': True, 'Export_boxplot': True} #Choose your preferemces of exporting files

ec = EvoCluster(
    optimizer,
    objective_func,
    dataset_list,
    num_of_runs,
    params,
    export_flags,
    auto_cluster=True,
    n_clusters='supervised',
    labels_exist=True,
    metric='euclidean'
)

ec.run() #run the framework
```
Now your experiment is ready to go. Enjoy!  

The results will be automaticly generated in a folder which is concatnated with the date and time of the experiment. this folder consists of three csv files and two types of plots:
- experiment.csv
- experiment_details.csv
- experiment_details_Labels.csv
- Convergence plot
- Box plot

## Datasets
The folder datasets in the repositoriy contains 30 datasets (All of them are obtained from Scikit learn, UCI, School of Computing at University of Eastern Finland, ELKI, KEEL, and Naftali Harris Blog).

To add new dataset:
- Put your dataset in a csv format (No header is required, labels are at the last column)
- Place the new datset files in the datasets folder.
- Add the dataset to the datasets list in the optimizer.py (Line 19).
  
## Citation Request:

Please include these citations if you plan to use this Framework:

- Qaddoura, Raneem, Hossam Faris, Ibrahim Aljarah, and Pedro A. Castillo. "EvoCluster: An Open-Source Nature-Inspired Optimization Clustering Framework." SN Computer Science, 2(3), 1-12, 2021.

- Qaddoura, Raneem, Hossam Faris, Ibrahim Aljarah, and Pedro A. Castillo. "EvoCluster: An Open-Source Nature-Inspired Optimization Clustering Framework in Python." In International Conference on the Applications of Evolutionary Computation (Part of EvoStar), pp. 20-36. Springer, Cham, 2020.

- Hossam Faris, Ibrahim Aljarah, Sayedali Mirjalili, Pedro Castillo, and J.J Merelo. "EvoloPy: An Open-source Nature-inspired Optimization Framework in Python". In Proceedings of the 8th International Joint Conference on Computational Intelligence - Volume 3: ECTA,ISBN 978-989-758-201-1, pages 171-177.

