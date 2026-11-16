# Notebooks

Run them in order. Each notebook explains its own steps and shows its results inline.

| # | Notebook | What it is for | Who wrote it |
|---|---|---|---|
| 00 | `00_data_pipeline.ipynb` | Class ids, label-format check, spot-check pictures, group split, box counts, sources | data team |
| 01 | `01_baseline_evaluation.ipynb` | How the pretrained model scores before our training | Captain |
| 02 | `02_train_model.ipynb` | Training the animal detector | Captain |
| 03 | `03_evaluate_and_compare.ipynb` | Scores of the trained models, side by side | Captain |
| 04 | `04_demo_inference.ipynb` | Running the detector on pictures and video | Captain |
| 05 | `05_web_app_guide.ipynb` | How the website works; person rule demo; log queries; start and stop the site | web team |

## How to run

1. Open the notebook in Jupyter with the `animal-det` kernel. Register it once with `python -m ipykernel install --user --name animal-det --display-name "Python (animal-det)"` (setup: root `README.md`), then start `jupyter lab`.
2. Run the cells from the top (**Run all**). The first cell finds the project root, so paths are relative to it.
3. The default is a **quick run** (about 10 images, or 10 test images for the rule demo). Each notebook has a parameter cell at the top with a `FULL_RUN` switch for the full run. Notebooks 00 and 05 do not train anything.
4. Heavy steps (training, many-image inference, the web server) run in a subprocess under a memory cap:
   `systemd-run --user --scope -p MemoryMax=15G -p MemorySwapMax=0`.

## Notes

- Nothing here installs packages (`pip install -r requirements.txt` is in the root `README.md`).
- `run_notebook.py` is only a fallback runner for environments without Jupyter; with Jupyter installed use `jupyter nbconvert --to notebook --execute --inplace <file>`.
- The web server listens on `127.0.0.1` only. Notebook 05 stops it at the end.
