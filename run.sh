# activate venv and set Python path
source ~/projects/venvs/XMTC-Baselines/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/XMTC-Baselines/

python main.py \
  tasks=[fit,predict,eval] \
  model=XR-TFMR\
  data=EURLEX57K \
  data.folds=[0]
