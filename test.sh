rm percepoutput.txt vanillamodel.txt averagedmodel.txt
python perceplearn.py data/train-labeled.txt
python percepclassify.py vanillamodel.txt data/dev-text.txt
python metric.py data/dev-key.txt percepoutput.txt

rm percepoutput.txt
python percepclassify.py averagedmodel.txt data/dev-text.txt
python metric.py data/dev-key.txt percepoutput.txt
