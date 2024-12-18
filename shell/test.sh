#!/bin/bash
python src/main.py --dataset=books --inference_only=1 --state_dict_path=./result/books/model/DART/books_DART.pth
python src/main.py --dataset=ml --inference_only=1 --state_dict_path=./result/ml/model/DART/ml_DART.pth
exit 0
 