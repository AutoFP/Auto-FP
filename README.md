# Auto-FP
This is the public repository of Auto-FP paper, which includes technical report, source code, datasets and experimental results.

## Datasets
You can download all 45 real-world datasets from [here](https://drive.google.com/drive/folders/1iW3pTC3swLDfaygIPQDVZ5egVUP8USqy?usp=sharing)

## Raw Experimental Results
You can find the raw experimental results [here](https://drive.google.com/drive/folders/1xeX9-2Ny0P2tF-N2UoUFSXddoLJFNoTb?usp=sharing)
These results are used for generating table and figures in paper and technical report.

## Dependencies
- numpy
- sklearn
- tpot
- joblib

## Code References
[ENAS](https://github.com/melodyguan/enas): reference for implementation of ENAS

[Hyperopt](https://github.com/hyperopt/hyperopt): for `Anneal` and `TPE`
* Change: add `output_dir` parameter to output pipeline pick-up time

[SMAC3](https://github.com/automl/SMAC3): for `SMAC`
* Change: add `output_dir` parameter to output pipeline pick-up time
