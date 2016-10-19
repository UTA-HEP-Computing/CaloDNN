
#PBS -V

cd /home/afarbin/Sherpa/DLKit/
source setup.sh

mkdir -p ScanLogs
output=ScanLogs/$PBS_ARRAYID.log

echo $output > $output

python -m MEDNN.Experiment &> $output



