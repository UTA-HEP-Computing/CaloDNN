#PBS -V

cd /home/afarbin/Sherpa/DLKit/
source setup.sh

python CaloDNN/LCDData.py &> Merge.log



