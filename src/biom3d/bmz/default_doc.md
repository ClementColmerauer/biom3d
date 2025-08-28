# Documentation
This is the default documentation for a Biom3d model in Bioimage.io.

## Training
This part is only if you did fine tunning

**Example:**  
This model is a fine-tuned version of the model `INSERT MODEL NAME OR ID`, available [here](INSERT URL).
It was fine-tuned using the dataset `My Dataset`, with the following characteristics:
- Specific modality (e.g., MRI, confocal)
- Imaging settings (e.g., 1 µm isotropic resolution)
- Any preprocessing applied (e.g., bias correction)

## Training data
This section describes the dataset used for training in detail, including:

- Source of the data (e.g., public dataset, internal)
- Image modalities and acquisition parameters
- Labeling protocol (manual annotation, automatic, etc.)
- Number of samples used

You may also include information extracted from the training configuration (`model log/config.yaml`):

- `MEDIAN_SPACING`: spacing for each spatial axis
- `CLIPPING_BOUNDS`: 0.05% / 99.5% intensity percentile clipping range
- `INTENSITY_MOMENTS`: mean and standard deviation of the voxel intensities

## Validation
This model was validated using the following metric: `INSERT METRIC NAME`.

Including validation details helps with reproducibility and guides users during fine-tuning.

🛈 *Note:*  
By default in Biom3d, evaluation (e.g., with `eval` functions) uses the **Dice coefficient**.


## Training schedule
Here you describe what parameter you used for training, if it isn't a fine tuning you don't need to separate it in `Initial training` and `Fine tuning`. Those parameter can be found in your model log/config.yaml.

### Initial training
- Learning rate: LR_START
- Weight decay: WEIGHT_DECAY
- Number of epochs: NB_EPOCHS
- Number of worker: NUM_WORKERS
- Size of batches: BATCH_SIZE
- LossFunction: TRAIN_LOSS
Or any other parameter you deem useful

### Fine tuning
The same kind as `Initial training`

## Contact
For questions or support, please contact:

- `Your Name <your.email@example.com>`
If you encounter an issue with a Biom3d model or Biom3d itself, [open an issue](https://github.com/GuillaumeMougeot/biom3d/issues)