# 🌿 Plant Leaf Disease Detector — CV Semester Project

A MobileNetV2 (transfer learning) model that classifies **15 plant leaf disease /
healthy classes** for bell pepper, potato and tomato, trained on the
**PlantVillage** dataset. Ships with a Streamlit web app and an auto-generated
project report.

## Project files

| File | Purpose |
|------|---------|
| `app.py` | **Streamlit web frontend** — upload a leaf, get diagnosis + treatment |
| `disease_model.py` | Model loading + single-image inference (used by FastAPI) |
| `disease_info.py` | Per-disease symptoms & treatment knowledge base |
| `main.py` | FastAPI REST API (`/predict-disease`) |
| `checkpoint_epoch4.pth` | Trained MobileNetV2 weights + class names |


## Run the web app

```bash
# makesure to have streamlit installed
streamlit run app.py
```

Then open http://localhost:8501, upload a leaf photo, and read the predicted
disease, confidence, top-3 chart and treatment advice.



## Model summary

- **Architecture:** MobileNetV2 (ImageNet-pretrained), classifier head replaced
  with `Dropout(0.2) + Linear(1280 → 16)`
- **Parameters:** ~2.24M · **Input:** 224×224×3 · **Checkpoint:** 9.2 MB
- **Classes:** 15 real classes + 1 spurious `split` folder artifact (flagged in
  the app; see report §6.6)
