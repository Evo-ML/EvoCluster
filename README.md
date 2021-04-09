### EvoCluster: An Open-Source Nature-Inspired Optimization Clustering Framework in Python

### Beta Version ###

EvoCluster is an open source and cross-platform framework implemented in Python which includes the most well-known and recent nature-inspired meta heuristic  optimizers  that  are  customized  to  perform  partitional  clustering tasks.  The  goal  of  this  framework  is  to  provide  a  user-friendly  and  customizable implementation of the metaheuristic based clustering algorithms which canbe utilized by experienced and non-experienced users for different applications.The framework can also be used by researchers who can benefit from the implementation of the metaheuristic optimizers for their research studies. EvoClustercan be extended by designing other optimizers, including more objective func-tions, adding other evaluation measures, and using more data sets. The current implementation of the framework includes ten metaheuristic optimizers, thirty datasets, five objective functions, twelve evaluation measures, more than twenty distance measures, and ten different ways for detecting the k value. The  source code of EvoCluster is publicly available at (http://evo-ml.com/evocluster/).

## Features
- Ten nature-inspired metaheuristic optimizers are implemented (SSA, PSO, GA, BAT, FFA, GWO, WOA, MVO, MFO, and CS).
- Five objective functions (SSE, TWCV, SC, DB, and DI).
- Thirty datasets obtained from Scikit learn, UCI, School of Computing at University of Eastern Finland, ELKI, KEEL, and Naftali Harris Blog
- Twelve evaluation measures (SSE, Purity,	Entropy,	HS,	CS,	VM,	AMI,	ARI,	Fmeasure,	TWCV,	SC,	Accuracy,	DI,	DB,	and Standard Diviation)
- More than twenty distance measures
- Ten different ways for detecting the k value
)
- The implimentation uses the fast array manipulation using [`NumPy`] (http://www.numpy.org/).
- Matrix support using [`SciPy`'s] (https://www.scipy.org/) package.
- Simple and efficient tools for prediction using [`sklearn`] (https://scikit-learn.org/stable/)
- File data analysis and manipulation tool using [`pandas`] (https://pandas.pydata.org/)
- Plot interactive visualizations using [`matplotlib`] (https://matplotlib.org/)
- More optimizers, objective functions, adatasets, and evaluation measures are comming soon.
 
 

## Installation
- Python 3.xx is required.

Run

    pip3 install -r requirements.txt

(possibly with `sudo`)

That command above will install  `NumPy`, `SciPy`, `sklearn`, `pandas`, and `matplotlib` for you.

- If you are installing EvoCluster onto Windows, please Install Anaconda from here https://www.continuum.io/downloads, which is the leading open data science platform powered by Python.
- If you are installing onto Ubuntu or Debian and using Python 3 then
  this will pull in all the dependencies from the repositories:
  
      sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev

## Get the source

Clone the Git repository from GitHub

    git clone https://github.com/RaneemQaddoura/EvoCluster.git

## Quick User Guide
EvoCluster Framework contains more than thirty datasets (Obtainied from UCI repository, scikit learn, School of Computing at University of Eastern Finland, ELKI, KEEL, and Naftali Harris). 
The main file is the main.py, which considered the interface of the framewok. In the optimizer.py you 
can setup your experiment by selecting the optmizers, the datasets, objective functions, number of runs, number of iterations, and the population size. The following is a sample example to use the EvoCluster framework.

Guding videos:
- Introduction video: https://www.youtube.com/watch?v=3DYIdxILZaw
- Demo video: https://www.youtube.com/watch?v=TOIo9WMBWUc

## Select framework parameters

Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS". For example:

        optimizer=["SSA","PSO","GA"]

Select objective function from the list of available ones:"SSE","TWCV","SC","DB","DI". For example:

        objectivefunc=["SSE","TWCV"] 

Select data sets from the list of available ones
The folder datasets in the repositoriy contains 30 datasets (All of them are obtained from Scikit learn, UCI, School of Computing at University of Eastern Finland, ELKI, KEEL, and Naftali Harris Blog).

To add new dataset:
- Put your dataset in a csv format (No header is required, labels are at the last column)
- Place the new datset files in the datasets folder.
- Add the dataset to the datasets list in the optimizer.py (Line 19).
  
For example, if the dastaset name is seed, the new line  will be like this:

        datasets=["aggregation", "seeds"]

Select number of repetitions for each experiment. 
To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.

        NumOfRuns=30

Select general parameters for all optimizers (population size, number of iterations) ....

        params = {'PopulationSize' : 30, 'Iterations' : 50}

Choose whether to Export the results in different formats

        export_flags = {'Export_avg':True, 'Export_details':True, 'Export_details_labels':True, 'Export_convergence':True, 'Export_boxplot':True}

run the framework

        run(optimizer, objectivefunc, dataset_List, NumOfRuns, params, export_flags)

Now your experiment is ready to go. Enjoy!  

The results will be automaticly generated ina folder which is concatnated with the date and time of the experiment. this folder consists of three csv files and two types of plots:
- experiment.csv
- experiment_details.csv
- experiment_details_Labels.csv
- Convergence plot
- Box plot

The experiment and the experiment_details files contain the following measures:

    Optimizer: The name of the used optimizer
    Dataset: The name of the dataset.
    objfname: The objective function/ Fitness function
    Experiment: Experiment ID/ Run ID.
    startTime: Experiment's starting time
    EndTime: Experiment's ending time
    ExecutionTime : Experiment's executionTime (in seconds)
    SSE
    Purity
    Entropy
    HS
    CS
    VM
    AMI
    ARI
    Fmeasure
    TWCV
    SC
    Accuracy
    DI
    DB
    STDev
    Iter1	Iter2 Iter3 Iter4... : Convergence values (The bjective function values after every iteration).	


## Contribute
- Issue Tracker: https://github.com/RaneemQaddoura/EvoCluster/issues  
- Source Code: https://github.com/RaneemQaddoura/EvoCluster

## Useful Links
- Web Page: http://evo-ml.com/evocluster/
- Introduction video: https://www.youtube.com/watch?v=3DYIdxILZaw
- Demo video: https://www.youtube.com/watch?v=TOIo9WMBWUc
- Paper: https://link.springer.com/chapter/10.1007/978-3-030-43722-0_2
- Colab: https://github.com/RaneemQaddoura/EvoCluster/blob/master/EvoCluster.ipynb

## Support

Use the [issue tracker](https://github.com/RaneemQaddoura/EvoCluster/issues). 

## Citation Request:

Please include these citations if you plan to use this Framework:

- Qaddoura, Raneem, Hossam Faris, Ibrahim Aljarah, and Pedro A. Castillo. "EvoCluster: An Open-Source Nature-Inspired Optimization Clustering Framework." SN Computer Science, 2(3), 1-12, 2021.

- Qaddoura, Raneem, Hossam Faris, Ibrahim Aljarah, and Pedro A. Castillo. "EvoCluster: An Open-Source Nature-Inspired Optimization Clustering Framework in Python." In International Conference on the Applications of Evolutionary Computation (Part of EvoStar), pp. 20-36. Springer, Cham, 2020.

- Ruba Abu Khurma, Ibrahim Aljarah, Ahmad Sharieh, and Seyedali Mirjalili. Evolopy-fs: An
open-source nature-inspired optimization framework in python for feature selection. In Evolutionary
Machine Learning Techniques, pages 131–173. Springer, 2020

- Hossam Faris, Ibrahim Aljarah, Sayedali Mirjalili, Pedro Castillo, and J.J Merelo. "EvoloPy: An Open-source Nature-inspired Optimization Framework in Python". In Proceedings of the 8th International Joint Conference on Computational Intelligence - Volume 3: ECTA,ISBN 978-989-758-201-1, pages 171-177.

- Ibrahim Aljarah, Majdi Mafarja, Ali Asghar Heidari, Hossam Faris, and Seyedali Mirjalili. Multiverse optimizer: Theory, literature review, and application in a data clustering. In Nature-Inspired Optimizers: Theories, Literature Reviews and Applications, pages 123–141. Springer International Publishing, Cham, 2020


