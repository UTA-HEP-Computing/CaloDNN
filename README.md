# CaloDNN

A DLKit package for Deep Neural Network Calorimetry tasks, such as
Classification, Energy Regression, and Generative Models. Currently
the package is configured to work on simulated data for the LCD
detector concept for the CLIC collider. 

## Installation
Get Data Provider Core
```bash
$ git clone https://gitlab.anomalousdl.com/open-source/data_provider_core
$ cd ./data_provider_core
$ pip install -e .
```


Get the packages:

    git clone https://bitbucket.org/anomalousai/DLKit
    cd DLKit
    git clone https://github.com/UTA-HEP-Computing/CaloDNN

Work from DLKit Directory:

    cd DLKit

Make sure CUDA is setup if you are using GPUs (default) or add `--cpu`
flag below to run on CPUs.

Make sure requirements are installed:

    pip install -r CaloDNN/requirements.txt

## Running

Currently only particle classification task is implemented. The
training and analysis is primarly performed by running an "experiment"
on the command line or through Jupyter notebooks (see examples in
package). Run the experiment with `--help` switch to see the command-line
options:

    python -m CaloDNN.ClassificationExperiment --help

Edit `CaloDNN/ClassificationScanConfig.py` to set input files and
experiment configuration. Please read the comments in the various
sections of `CaloDNN/ClassificationExperiments.py` for more details
about reading input files and configuring the experiment and
hyperparameter scans.

Run an experiment by:

    python -m CaloDNN.ClassificationExperiment

Run a series of experiments with different parameters (as configured
in `Params` dictionary in `CaloDNN/ClassificationScanConfig.py`) by
first seeing how many possible experiments:

    python -m CaloDNN.ClassificationScanConfig

and then running N experiments with the different configurations using
`-s` option and an integer between 0 and N:

    python -m CaloDNN.ClassificationExperiment -s 5

The models and results are stored in `TrainedModels` directory.

For example, you can get a summary of results (all numbers stored in
the Model MetaData) using:

    python -m DLAnalysis.ScanAnalysis TrainedModels/

You can load a trained model:

    python -im DLAnalysis.LoadModel TrainedModels/<ModelName>
       
or all of your trained models:

    python -im DLAnalysis.LoadModel TrainedModels/*

and use the model for inference, further training, or inspection of
metadata (e.g. using `MyModel[0].MetaData`).

After a scan, examine the progress and status of all of your trained
models, by copying the example Jupyter notebook, running Jupyter, and
editing as needed:

    cp CaloDNN/AnalyzeScan.ipynb .
    jupyter notebook AnalyzeScan.ipynb 

Or analyze the performance of a specific model, looking at ROC curves or
energy dependence of the performance:

    cp CaloDNN/AnalyzePerformace.ipynb .
    jupyter notebook AnalyzePerformance.ipynb 

## Notes for running on batch (for example on on UTA-DL Cluster)

For running in batch, edit `setup.sh` to properly setup your
environment. Use the PBS submit script `ScanJob.sh` as a model for
submitting to your system. If you have torque/PBS running, simply edit
`ScanJob.sh` to point to right path and then:

    qsub -q <queuename> -t 0-19 CaloDNN/ScanJob.sh

to (for example) submit 20 jobs, each with different configuration.
On the HEP-DL cluster, use `gpu_queue` as the queue. Note that the GPU
used is determined by the enviroment variables set by torque. See
`ClassificationArguments.py` for details. You may need to adjust this
mechanism for your site.

If you use the `GracefulExit` callback (comment/uncomment line in
`ClassificationExperiment.py`), you can gracefully stop your running
jobs by delaying the SIGTERM and SIGKILL signals. For example:

    qdel -W 1200 all

will send a SIGTERM signal to job, which will cause it to stop
training and exit normally at end of current epoch. It will set a
SIGKILL signal after 20 minutes, in case some job takes too long to
stop.

Note the CaloDNN jobs are typically configured to gracefully end
training after a certain time period. If you resubmit a job, it will
automatically restart at the end of the last successful training run.

You can see the progress of running jobs (assuming common file
system):

    tail -f ScanLogs/*

## Notes for running Jupiter on UTA-DL Cluster

For simplicity, run Jupyter on the gateway. Use the --no-browser
switch to not start the browser from the server, and specify a port
(your choice):

    jupyter notebook AnalyzeScan.ipynb --no-browser --port=8888

You'll have to use ssh-tunnel to see the Jupyter server (on your local
system):

    ssh -NfL 8888:127.0.0.1:8888 orodruin.uta.edu

Replace 8888 with the Jupyter server port. Navigate you browser to
127.0.0.1:8888.

