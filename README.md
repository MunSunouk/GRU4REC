# Environment
- **PyTorch 0.4.0**
- Python 3.6.4
- pandas 0.22.0
- numpy 1.14.0
---

# Usage

## Training/Testing using Jupyter Notebook
See `example.ipynb` for the full jupyter notebook script that
1. Loads the data
2. Trains & tests a GRU4REC model
3. Loads & tests a pretrained GRU4REC model

## Training & Testing using `run_train.py`
- `run_train.py`

```
Test result: loss:0.019/recall:0.247/precision:0.617/mrr:0.169/time:0.007

```
## 이탈

```
$ python run_train.py > logs/train.out


```

