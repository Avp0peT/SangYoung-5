# Human Parsing (Human Part Segmentation)

This repository implements a human parsing / human part segmentation pipeline using a U-Net model built with
`segmentation_models_pytorch`.

This README explains how to run inference on a single image or a folder of images using the `--process` option
in `main.py`.

## Requirements

Python 3.7+ and the packages listed in `requirements.txt`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Quick usage — single image

Process a single image and save visualization, mask and extracted parts:

```powershell
python main.py --process --input C:\path\to\image.jpg
```

This will:
- load a model checkpoint (by default it looks for `checkpoints/final_model.pth`, `checkpoints/best_model.pth`, or the newest `.pth` in `checkpoints/`)
- create an output folder under `static/results/<timestamp>_<id>/` and save per-image outputs there

To explicitly specify a model checkpoint:

```powershell
python main.py --process --input C:\path\to\image.jpg --model project\checkpoints\final_model.pth
```

To disable the visualization images (only masks and extracted parts will be saved):

```powershell
python main.py --process --input C:\path\to\image.jpg --no-visualize
```

## Quick usage — folder (batch)

Process all images in a folder (recursively):

```powershell
python main.py --process --input C:\path\to\image_folder
```

Each image will be processed and a subfolder created per image under the run folder (e.g. `static/results/20251028_123456_ab12cd34/<image_name>/`).

## Integration with existing code

- `inference.py` provides the `HumanParsingInference` class with helpers for preprocessing, prediction, visualization, and part extraction.
- Use `python main.py --web` to run the Flask web app.

## Notes

- Tests in the repository were disabled / removed to streamline inference flow. If you want tests restored, we can add a pytest-based test suite.
- For GPU execution, ensure CUDA and a compatible PyTorch build are installed and pass `--device cuda` if automatic detection doesn't select it.

Run `python main.py --help` for full options.
