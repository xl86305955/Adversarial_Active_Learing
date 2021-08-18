All:
	python run_experiment.py halfmoon nn FGSM 100

halfmoon:
	python run_experiment.py halfmoon nn FGSM 100
	python run_experiment.py halfmoon nn DEEPFOOL 100
	python run_experiment.py halfmoon nn PGD 100
	python run_experiment.py halfmoon nn FGSM 250
	python run_experiment.py halfmoon nn DEEPFOOL 250
	python run_experiment.py halfmoon nn PGD 25
	python run_experiment.py mnist nn FGSM 100
	python run_experiment.py mnist nn DEEPFOOL 100
	python run_experiment.py mnist nn PGD 100
	python run_experiment.py mnist nn FGSM 250
	python run_experiment.py mnist nn DEEPFOOL 250
	python run_experiment.py mnist nn PGD 25
	python run_experiment.py halfmoon nn CW 100
	python run_experiment.py halfmoon nn CW 250

abalone:
	python run_experiment.py abalone nn FGSM 20
	python plot.py abalone FGSM 
	python run_experiment.py abalone nn DEEPFOOL 20
	python plot.py abalone DEEPFOOL
	python run_experiment.py abalone nn PGD 20
	python plot.py abalone PGD

mnist:
	python run_experiment.py mnist nn FGSM 20
	python run_experiment.py mnist nn FGSM 50
	python run_experiment.py mnist nn FGSM 500
	python run_experiment.py mnist nn PGD 20
	python run_experiment.py mnist nn PGD 50
	python run_experiment.py mnist nn PGD 500
	python run_experiment.py mnist nn DEEPFOOL 20
	python run_experiment.py mnist nn DEEPFOOL 50
	python run_experiment.py mnist nn DEEPFOOL 500
	python run_experiment.py mnist nn CW 20
	python run_experiment.py mnist nn CW 50
	python run_experiment.py mnist nn CW 500

