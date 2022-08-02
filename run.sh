# activate venv and set Python path
source ~/projects/venvs/XMTC-Baselines/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/XMTC-Baselines/

# XR-TFMR EURLEX57K
python main.py \
  tasks=[predict] \
  model=XR-TFMR \
  data=EURLEX57K \
  data.folds=[0]

