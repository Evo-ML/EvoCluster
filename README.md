### EvoCluster: An Open-Source Nature-InspiredOptimization Clustering Framework in Python

### Beta Version ###

EvoCluster is an open source and cross-platform framework imple-mented in Python which includes the most well-known and recent nature-inspiredmetaheuristic  optimizers  that  are  customized  to  perform  partitional  clustering tasks.  The  goal  of  this  framework  is  to  provide  a  user-friendly  and  customiz-able implementation of the metaheuristic based clustering algorithms which canbe utilized by experienced and non-experienced users for different applications.The framework can also be used by researchers who can benefit from the imple-mentation of the metaheuristic optimizers for their research studies. EvoClustercan be extended by designing other optimizers, including more objective func-tions, adding other evaluation measures, and using more data sets. The current implementation  of  the  framework  includes  ten  metaheristic  optimizers,  thirtydatasets,  five  objective  functions,  and  twelve  evaluation  measures.  The  source code of EvoCluster is publicly available at (http://evo-ml.com/2019/10/25/evocluster/).



The full list of implemented optimizers is available here https://github.com/7ossam81/EvoloPy/wiki/List-of-optimizers


## Features
- Ten nature-inspired metaheuristic optimizers are implemented.
- The implimentation uses the fast array manipulation using [`NumPy`] (http://www.numpy.org/).
- Matrix support using [`SciPy`'s] (https://www.scipy.org/) package.
- More optimizers are comming soon.
 

## Installation
- Python 3.xx is required.

Run

    pip3 install -r requirements.txt

(possibly with `sudo`)

That command above will install  `sklearn`, `NumPy` and `SciPy` for
you.

- If you are installing EvoCluster onto Windows, please Install Anaconda from here https://www.continuum.io/downloads, which is the leading open data science platform powered by Python.
- If you are installing onto Ubuntu or Debian and using Python 3 then
  this will pull in all the dependencies from the repositories:
  
      sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev

## Get the source

Clone the Git repository from GitHub

    git clone https://github.com/RaneemQaddoura/EvoCluster.git


## Quick User Guide
EvoCluster Framework contains more than thirty datasets (Obtainied from UCI repository, scikit learn, School of Computing at University of Eastern Finland, ELKI, KEEL, and Naftali Harris). 
The main file is the main.py, which considered the interface of the framewok. In the main.py you 
can setup your experiment by selecting the optmizers, the datasets, number of runs, number of iterations, number of neurons
and population size. The following is a sample example to use the EvoCluster framework.
To choose PSO optimizer for your experiment, change the PSO flag to true and others to false.

Select optimizers:    
PSO= True  
MVO= False  
GWO = False  
MFO= False  
.....


After that, Select datasets:

datasets=["aggregation.csv", "seeds.csv"]

The folder datasets in the repositoriy contains 3 binary datasets (All of them are obtained from UCI repository).

To add new dataset:
- Put your dataset in a csv format (No header is required)
- Normalize/Scale you dataset ([0,1] scaling is prefered) #(Optional)
- Place the new datset files in the datasets folder.
- Add the dataset to the datasets list in the main.py (Line 18).
  
  For example, if the dastaset name is seed, the new line  will be like this:
        
        datasets=["aggregation.csv", "seeds.csv"]


Change NumOfRuns, PopulationSize, and Iterations variables as you want:
    
    For Example: 

    NumOfRuns=10  
    PopulationSize = 50  
    Iterations= 1000

Now your experiment is ready to go. Enjoy!  

The results will be automaticly generated in excel file called Experiment which is concatnated with the date and time of the experiment.
The results file contains the following measures:


    Optimizer	Dataset	objfname	Experiment	startTime	EndTime	ExecutionTime	trainAcc	testAcc
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
    Iter1	Iter2 Iter3... : Convergence values (The bjective function values after every iteration).	


## Contribute
- Issue Tracker: https://github.com/RaneemQaddoura/EvoCluster/issues  
- Source Code: https://github.com/RaneemQaddoura/EvoCluster

## Support

Use the [issue tracker](https://github.com/RaneemQaddoura/EvoCluster/issues). 

## Citation Request:

Please include these citations if you plan to use this Framework:

- Ruba Abu Khurma, Ibrahim Aljarah, Ahmad Sharieh, and Seyedali Mirjalili. Evolopy-fs: An
open-source nature-inspired optimization framework in python for feature selection. In Evolutionary
Machine Learning Techniques, pages 131–173. Springer, 2020

- Hossam Faris, Ibrahim Aljarah, Sayedali Mirjalili, Pedro Castillo, and J.J Merelo. "EvoloPy: An Open-source Nature-inspired Optimization Framework in Python". In Proceedings of the 8th International Joint Conference on Computational Intelligence - Volume 3: ECTA,ISBN 978-989-758-201-1, pages 171-177.

- Raneem Qaddoura, Hossam Faris, and Ibrahim Aljarah*. An efficient clustering algorithm based
on the k-nearest neighbors with an indexing ratio. International Journal of Machine Learning and Cybernetics, pages 1–40, 2019.

- Ibrahim Aljarah, Majdi Mafarja, Ali Asghar Heidari, Hossam Faris, and Seyedali Mirjalili.
Clustering analysis using a novel locality-informed grey wolf-inspired clustering approach. Knowledge and Information Systems, pages 1–33, 2019.

- Sarah Shukri, Hossam Faris, Ibrahim Aljarah*, Seyedali Mirjalili, and Ajith Abraham. Evolutionary static and dynamic clustering algorithms based on multi-verse optimizer. Engineering Applications of Artificial Intelligence, 72:54–66, 2018. 

- Ibrahim Aljarah, Majdi Mafarja, Ali Asghar Heidari, Hossam Faris, and Seyedali Mirjalili. Multiverse optimizer: Theory, literature review, and application in a data clustering. In Nature-Inspired Optimizers: Theories, Literature Reviews and Applications, pages 123–141. Springer International Publishing, Cham, 2020


