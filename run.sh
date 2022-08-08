# activate venv and set Python path
source ~/projects/venvs/XMTC-Baselines/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/XMTC-Baselines/

# XR-TFMR Wiki10-31k
python main.py \
  tasks=[preprocess,fit,predict] \
  model=XR-TFMR-Wiki10-31k \
  data=Wiki10-31k \
  data.folds=[1,2,3,4]

