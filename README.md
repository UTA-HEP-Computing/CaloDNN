# CaloDNN

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

Check out the arguments:

    python -m CaloDNN.ClassificationExperiment --help

Edit `CaloDNN/ClassificationScanConfig.py` to set input files and
experiment configuration.

Run an experiment:

    python -m CaloDNN.ClassificationExperiment

Run a series of experiments with different parameters (as configured
in `Params` dictionary in `CaloDNN/ClassificationScanConfig.py`) by first seeing
how many possible experiments:

    python -m CaloDNN.ClassificationScanConfig

and then running N experiments with the different configurations using
`-s` option and an integer between 0 and N:

    python -m CaloDNN.ClassificationExperiment -s 5

The models and results are stored in `TrainedModels` directory.

For example, you can get a summary of results (all numbers stored in
the Model MetaData) using:

    python -m DLTools.ScanAnalysis TrainedModels/

You can load a trained model:

    python -im TrainedModels/<ModelName>
       
or all of your trained models:

    python -im TrainedModels/*

and use the model for inference, further training, or inspection of
metadata (e.g. using `MyModel[0].MetaData`).

For running in batch, edit `setup.sh` to properly setup your
environment. Use the PBS submit script `ScanJob.sh` as a model for
submitting to your system. If you have torque/PBS running, simply
edit `ScanJob.sh` to point to right path and then:

     qsub -q <queuename> -t 0-44 CaloDNN/ScanJob.sh

to (for example) submit 45 jobs, each with different configuration. On
the UTA-DL cluster, use `gpuqueue` as the queue. Note that the GPU
used is determined by the enviroment variables set by torque. See
`ClassificationArguments.py` for details. You may need to adjust this
mechanism for your site.
