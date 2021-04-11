# IGM Stock Forecast And Trades
### Two type of ways to pass parameters.
```
usage: trader.py [-h] [-p | -t] [--model_input MODEL_INPUT]
                 [--train_input TRAIN_INPUT] [--test_input TEST_INPUT]
                 [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -p, --predict         predict electricity
  -t, --training        training model
  --model_input MODEL_INPUT
                        input model file name
  --train_input TRAIN_INPUT
                        input training data file name
  --test_input TEST_INPUT
                        input testing data file name
  --output OUTPUT       output file name

```

* Training model with the input csv file
```python
#Example
python trader.py -t --train_input ".\training.csv" --test_input "./testing.csv" --output "model1.h5"
```

```python
#Example
python trader.py -p --train_input ".\training.csv" --test_input "./testing.csv" --output "output.csv" --model_input ".\model1.h5"
```