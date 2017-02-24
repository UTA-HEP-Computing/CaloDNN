#PBS -V
cd <Place path here>
source setup.sh

mkdir -p ScanLogs
output=ScanLogs/$PBS_ARRAYID.log

echo $output > $output

python -m CaloDNN.ClassificationExperiment -s $PBS_ARRAYID &> $output



