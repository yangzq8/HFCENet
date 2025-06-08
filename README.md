## Environment
python 3.7

torch 1.5.1 + cu101

Open3D 0.10.0.1

spicy 1.4.1

hyp5 2.10.0

tqdm 4.47.0

## Data

For ModelNet40, the data will be downloaded automatically. If not, please download from: https://pan.baidu.com/s/11dyVn0l0LMnr0QwszVY8OQ pwd: mkm7

## Test
The easiest way to run the code is using the following command
```
python test.py --model_trained_path your_path
```

## Model_trained

clean: https://pan.baidu.com/s/1SV5ZSdskbqMhpIwwsp1G4Q password: d82g

## Train

The easiest way to run the code is using the following command
```
python main.py --exp_name exp
```
This command will run an experiment on the ModelNet40 dataset (automatically downloaded) with all the options set to default. You can see at the end of main.py a list of options that can be used to control hyperparameters of the model and experiment settings. The comments in the file should be enough to understand them.

