#PBS -V

cd /home/wghilliard/tmp/DLKit
source setup.sh

mkdir -p ScanLogs
output=ScanLogs/$PBS_ARRAYID.log

echo $output > $output

nvidia-docker run -v /data:/data -v TrainedModels:/opt/DLKit/TrainedModels wghilliard/dlkit_cdnn:v0 bash -c "cd /opt/DLKit && python -m CaloDNN.ClassificationExperiment.py" &> $output
