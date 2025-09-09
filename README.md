# MALDI-ST - external validation instruction


## System requirements and Installation

Create conda environment

```
conda env create -f envs/maldi_st.yml
```

## Preprocess MALDI-TOF mass spectra
You will need to provide a CSV input file (`input_file_path`) that includes the isolate ID (column `id`) and the corresponding Bruker run path (column `spot_dir`) for each isolate. Please also specify the output directory (`outdir`) where the preprocessed spectra should be saved. An example of the input file can be found [here](https://drive.google.com/file/d/1IRBgP25pUrJXit31VOzBheZFvEd0hfus/view?usp=drive_link)
```
Rscript scripts/preprocess.R input_file_path outdir
```

### Preparing label files
Currently, the model is limited to predicting only a subset of common STs for each species. These include:

> - *Escherichia coli* (8): ST12, ST38, ST69, ST73, ST95, ST127, ST131, ST1193
> - *Pseudomonas aeruginosa* (5): ST111, ST244, ST274, ST649, ST788
> - *Staphylococcus aureus* (7): ST5, ST12, ST15, ST22, ST30, ST45, ST398
> - *Enterococcus faecium* (4): ST78, ST80, ST796, ST1424

The label input file must include two columns: the `id` column and the `ST` column. An example of the label file can be found [here](https://drive.google.com/file/d/1bL6-_WKnh_An9Q0cbkC8ICLkApRbEV2t/view?usp=drive_link).

### Run prediction
First, you will need to download the model weights which can be find [here](https://drive.google.com/file). Please extract and place them in the current directory.

```
conda activate maldi_st

for run_id in {0..9}; do
python ../../src/predict.py \
    -cp "${PWD}" \
    data.run_id="${run_id}" \
done
```

