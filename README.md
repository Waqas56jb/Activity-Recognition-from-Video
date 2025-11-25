# Human Activity Recognition from Video

Human Activity Recognition (HAR) system that classifies 15 everyday actions from RGB frames and video clips. The project fine‑tunes EfficientNet‑B3 on a balanced dataset of 12 600 labeled images, then reuses the trained model for frame‑by‑frame video inference.

## Dataset
- Metadata: `Dataset/Training_set.csv` and `Dataset/Testing_set.csv` (filename ↔ activity label).
- Images: `Dataset/train/` and `Dataset/test/`, each containing 15 balanced activity folders (calling, dancing, hugging, etc.).
- The training notebook performs an 80/20 split using `ImageDataGenerator`, so no manual split needed.

> **Note:** Dataset files are ignored via `.gitignore` because of their size. Place them under the paths above before training.

## Training (`train.ipynb`)
1. Install dependencies: `pip install -r requirements.txt`.
2. Launch Jupyter and open `train.ipynb`.
3. Run cells sequentially:
   - Load & visualize dataset.
   - Build EfficientNet‑B3 classifier head.
   - Stage 1: freeze base, train 25 epochs (Adam 1e‑4).
   - Stage 2: unfreeze last 30 layers, fine‑tune 10 epochs (Adam 1e‑5→2e‑6).
4. Best weights are saved automatically (`best_model.keras`). Notebook reloads them and exports `activity_recognition_model.keras` plus `class_names.pkl`.
5. Evaluation graphics (`training_history.png`, `confusion_matrices_comparison.png`, etc.) are saved to the repo root for reporting.

## Inference (`test.py`)
1. Ensure `activity_recognition_model.keras` and `class_names.pkl` exist in the project root.
2. Run `python test.py` and provide the path to a video file when prompted.
3. The script:
   - Reads each frame with OpenCV.
   - Resizes to 160×160, applies EfficientNet preprocessing.
   - Predicts the activity and overlays the label/confidence (≥30 %).
   - Logs results with timestamps in `result.txt`.

## Artifacts
- `activity_recognition_model.keras`: final model for deployment.
- `best_model_weights.weights.h5`, `best_stgcn_model.keras`: additional checkpoints/experiments (not used in `test.py`).
- `Artifacts/` and `result/`: example frames, plots, and screenshots generated post-training or during demonstrations.

## Repository Structure
```
├── Dataset/                 # Training & testing images + CSV labels (ignored)
├── train.ipynb              # End-to-end training & evaluation pipeline
├── test.py                  # Video inference script
├── activity_recognition_model.keras / class_names.pkl
├── requirements.txt
├── Artifacts/, result/, *.png  # Generated visuals (ignored by Git)
└── README.md
```

## Reproducibility Notes
- Tested with TensorFlow ≥2.10, Python ≥3.10.
- GPU acceleration strongly recommended; adjust batch size if VRAM is limited.
- Modify `ImageDataGenerator` parameters in `train.ipynb` to experiment with augmentation strength.
- To evaluate on a held-out test CSV, adapt the validation workflow in section 8 of the notebook.

## Future Improvements
- Integrate temporal models (3D CNN, Transformer, ST-GCN) for better motion understanding.
- Apply test-time augmentation or multi-crop inference to boost accuracy.
- Build a lightweight GUI/REST API around `test.py` for end users.

For questions or contributions, open an issue or reach out to the maintainer.

