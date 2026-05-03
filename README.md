# Blood Cell Classification (Shallow CNN) — PyTorch

Classifies **3 types** from microscopic images:

- **RBC** (Red Blood Cells)
- **WBC** (White Blood Cells)
- **Platelets** (folder can be `Platelets` / `Platelet` / `Platlets` — any name is fine; the model predicts exactly the folder name)

## Step-by-step execution order (first → last)

### Step 1 — Install (first)

```bash
cd blood-cell-cnn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2 — Put dataset in correct place

Recommended structure:

```
blood-cell-cnn/
  data/
    train/
      RBC/
      WBC/
      Platelets/
    val/
      RBC/
      WBC/
      Platelets/
    test/               # optional
      RBC/
      WBC/
      Platelets/
```

If your dataset is currently only:

```
my_dataset/
  RBC/
  WBC/
  Platlets/
```

Create `data/train|val|test` automatically:

```bash
python -m src.split_dataset --in_dir "C:\\path\\to\\my_dataset" --out_dir data --overwrite
```

### Step 3 — Train shallow CNN

```bash
python -m src.train --data_dir data --backbone shallow --epochs 20 --batch_size 64 --img_size 96
```

This prints a run folder like: `runs/20260331-153000`

### Step 4 — Predict from terminal (optional)

```bash
python -m src.predict --run_dir runs/<timestamp> --image "C:\\path\\to\\image.jpg"
```

### Step 5 — Upload-button UI (last)

```bash
streamlit run src/app_streamlit.py
```

Paste your run folder (example: `runs/20260331-153000`) and upload an image.

