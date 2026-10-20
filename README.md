# Bankruptcy prediction

Open `main.ipynb` and run the cells in order. Keep `data.csv` in the same folder.
The notebook includes saved results and plots.

Install the packages if needed:

```bash
python -m pip install -r requirements.txt
```

The project explores correlations, selects variables with L1-penalized logistic
regression, and compares decision trees with and without selection.
Average precision is the main metric because bankruptcy is rare in this dataset.

Source: [UCI Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction).

The course also requires a separate PDF report, including a full transcript of
the relevant LLM prompts and answers in an appendix.
