# SLIP
ZERO SHOT CONTEXT-BASED OBJECT SEGMENTATION USING SLIP (SAM+CLIP)

## Our Proposed Architecture

![Alt text](arch.png)

## Repository Structure

- `SLIP demo/`
  - `zero_shot_finetuned.ipynb` SLIP - Zero shot segmentation demo after finetuning CLIP.
  - `zero_shot_pretrained.ipynb` - SLIP - Zero shot segmentation demo using pretrained CLIP.
- `assests` - Contains images for plots, model architecture, and test images.
- `baseline classifier/`
  - `classifier output/`
    - `ResNet18_pokemon_output` - text file - output after training ResNet18 on pokemon dataset.
    - `VGG_pokemon_output` - text file - output after training VGG on pokemon dataset.
  - `models/`
    - `ResNet18.py` - ResNet18 model.
     - `VGG.py` - VGG model.
   - `run_resnet.sbatch` - script to train ResNet
   - `run_vgg.sbatch` - script to train vgg
- `evaluation/`
  - `ResNet_eval.ipynb` - ResNet evaluation on pokemon dataset.
  - `SLIP_segment_eval.ipynb` SLIP - Evalution of SLIP after finetuning CLIP, on pokemon dataset.
  - `make_evalutaion_dataest.py` Creates evaluation dataset.
  - `pokedex.csv` Contains information mapping image index to image class.
  - `pretrained_eval_segment.ipynb` SLIP - Evalution of SLIP using pretrained CLIP, on pokemon dataset.
- `finetuned CLIP/`
  - `captions.csv` - contains captions for CLIP finetuning. 
  - `clip_grid_search.py` - Runs grid search on CLIP for hyperparameter tuning.
  - `clip_grid_search_output` - contains output after running gridsearch.
  - `convert_txt_to_csv.py` - converts captions text file to a csv file.
  - `generate_captions.py` - Generates captions for pokemon dataest.
  - `run.sbatch` - script for running grid search.
- `plots/`
  - `plot_resnet.ipynb` - plots for resnet.
  - `plot_CLIP.ipynb` - plots for CLIP.
  - `text_for_plot.txt` - best CLIP model output during grid search.
