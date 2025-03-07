

## Environment
The requirements.txt file includes the required libraries for this project.

	python -m venv ynet
	source ./ynet/bin/activate
	pip install -r requirements.txt

## Datasets Downloading and Preproccesing

Downloads the dataset, creates the required data directories and preprocesses the data:

    sh data_download_and_preprocess.sh

## Model Evaluation
Evaluate the pre-trained models:

    python eval.py
    
This will report the quantitative comparison between ours and Y-Net + FFC and save the qualitative comparison to "./figs".
To compare the model parameters, set --print_params to True.


## Model Training
Train the Y-net + FFC model:

    python train.py --dataset Duke

Train the plain ours model:

    python train.py --model_name reynet_ffc --dataset Duke
