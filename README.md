# PHS-Net

This repository contains the implementation of **PHS-Net: Convolution and Transformer Dual-path with Selective Feature Fusion for Medical Image Segmentation**. It includes model code, training and inference scripts.

> **[PHS-Net: Convolution and Transformer Dual-path with Selective Feature Fusion for Medical Image Segmentation](https://iopscience.iop.org/article/10.1088/2631-8695/adebe0)**</br>
> Viet-Thanh Nguyen, Van-Truong Pham and Thi-Thao Tran</br>
> *Accepted at Engineering Research Express, 2025*

## Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/vietthanh2710/phs-net.git
    cd phs-net
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Training

1. **Prepare your data**  
   - Place your data as a `.npy` file with keys `"image"` and `"mask"`, or modify `train.py` to fit your data format.

2. **Edit `config.yaml`**  
   - Set hyperparameters such as `epochs`, `batch_size`, `learning_rate`, `loss_type`, and `save_dir`.

   Example:
   ```yaml
   epochs: 100
   batch_size: 8
   learning_rate: 0.001
   loss_type: "DiceTverskyLoss"
   dice_weight: 0.5
   save_dir: "checkpoints"
   ```

3. **Run training**
    ```bash
    python train.py
    ```

   - The best model will be saved to the directory specified by `save_dir`.

---

## Inference

1. **Run inference on a single image**
    ```bash
    python inference.py --image_path path/to/image.png --model_path checkpoints/best_model.pth --output_path mask.png
    ```

    - Supports `.jpg`, `.png`, and `.npy` images.
    - The output mask will be saved as a PNG file.

2. **Arguments**
    - `--image_path`: Path to the input image (required)
    - `--model_path`: Path to the trained model weights (required)
    - `--output_path`: Path to save the output mask (optional; if not set, mask will be displayed)
    - `--input_size`: Input size for the model, e.g., `--input_size 256 256` (default: 256x256)

---

## Citation

If you use this code for your research, please cite the original paper:

```bibtex
@article{10.1088/2631-8695/adebe0,
	author={Nguyen, Viet-Thanh and Pham, Van-Truong and Tran, Thi-Thao},
	title={PHS-Net: Convolution and Transformer Dual-path with Selective Feature Fusion for Medical Image Segmentation},
	journal={Engineering Research Express},
	url={http://iopscience.iop.org/article/10.1088/2631-8695/adebe0},
	year={2025},
}
```
