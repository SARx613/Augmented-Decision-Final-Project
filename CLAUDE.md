# Build a beginner-friendly variable-selection notebook

## 1. Your assignment

Build a new Jupyter notebook named `variable_selection.ipynb` at the root of the project described below. This notebook prepares and studies L1-penalized logistic regression as a variable selector for a future decision-tree classifier of Taiwanese bankruptcy.

Implement only variable selection, its supporting data preparation, and its diagnostics. Do not train a decision tree, compare classifier performance, select an alert threshold, or choose a final regularization parameter C. Those decisions belong to the following project stage.

The agreed approach is to examine `C = [0.01, 0.1, 1.0, 10.0]`, retain all four as candidates, and eventually choose C through validation of the complete selection-plus-tree procedure. This notebook must end with a detailed Markdown explanation of that next stage, specified at the end of this file.

Assume no previous project work has been completed. Establish the reproducible data split and cross-validation folds here. An existing `bankruptcy_prediction.ipynb` is not evidence of completed work and must not be modified.

This is a university project. Favor the simplest correct implementation that the students can understand, explain, and reproduce. The goal is methodological clarity, not production engineering or a large software framework.

## 2. Expected folder structure and allowed changes

This instruction file is intended to be placed at `FinalProject/CLAUDE.md`.

```text
FinalProject/
├── .venv/
├── context/
│   ├── CourseSlides/
│   │   ├── 1. From data to decisions - Losses, risks and decision rules.pdf
│   │   ├── 2. Statistical learning and classification - Empirical risk, generalization and minimax limits.pdf
│   │   ├── 3. Gaussian regression and variable selection - From sparse estimation to choosing useful variables.pdf
│   │   ├── 3. Statistical learning and classification - Decision trees using weak learners.pdf
│   │   └── 4. Regularization and variable selection - Ridge, lasso and the same warehouse prediction task.pdf
│   └── ProjectRequirements.pdf
├── data/
│   ├── data.csv
│   └── README.md
├── .gitignore
├── bankruptcy_prediction.ipynb
├── README.md
├── requirements.txt
└── CLAUDE.md
```

Create:

```text
FinalProject/
├── variable_selection.ipynb
└── outputs/
    └── variable_selection/
        ├── run_manifest.json
        ├── split_assignments.csv
        ├── selection_fits.csv
        ├── feature_selection.csv
        └── candidate_summary.csv
```

Inspect the actual filenames and existing READMEs and requirements before starting. Respect any applicable repository instructions. Do not rename source files to match an assumption in this document. If a lecture filename differs slightly, identify it by its title.

Preserve the existing notebook, source CSV, PDFs, and READMEs. Update `requirements.txt` only if a required dependency is absent, preserving its existing conventions and constraints. Do not replace an existing environment or unnecessarily upgrade installed packages.

Use `pathlib.Path` and project-relative paths. Expect the notebook to run with the project root as its working directory. If `data/data.csv` is not found, raise an informative error explaining the expected working directory; do not recursively search the computer or silently load another dataset.

## 3. Teaching and writing requirements

Write the notebook in English for a reader with almost no prior machine-learning knowledge.

- Use short, numbered sections following the sequence below.
- Every Python cell must be immediately preceded by a Markdown cell explaining its purpose, its inputs, and what its output will mean. Never place two Python cells consecutively.
- After a substantial table or figure, add a short Markdown interpretation before the next Python cell. This can be combined with the next section's explanation.
- Explain a term before using it: predictor, target, coefficient, scaling, regularization, validation fold, and data leakage.
- Introduce intuition before equations. Explain every symbol that appears in an equation.
- Keep Python cells short and focused. Avoid a single large cell containing the whole experiment.
- Use descriptive variable names and straightforward loops. Prefer two or three small functions to custom classes, a bespoke estimator, or a complicated package structure.
- Do not hide the main learning steps inside an unexplained pipeline. Explicit training-only `fit` steps are acceptable and desirable here.
- Avoid repeated prose, long theoretical proofs, giant dataframe dumps, and unexplained diagnostic output.
- Clearly mark illustrations as hypothetical and computed findings as results from the actual run.
- Describe results honestly. If regularization fails to remove many variables, or different folds select different predictors, report that outcome.

The final cell must be the detailed next-stage Markdown handoff. No executable cell may appear after it.

## 4. Scientific scope and commitments

The target is `Bankrupt?`, with 1 meaning bankruptcy and 0 meaning nonbankruptcy. Predictors are the other supplied columns.

The selector is L1-penalized logistic regression. It fits a binary-outcome model whose penalty can make some predictor coefficients zero. We retain predictors with nonzero coefficients, subject to the fixed numerical tolerance below, for the future tree.

The logistic model is a tool for selecting inputs. It is not the final classifier in this project. Do not report its accuracy, balanced accuracy, ROC-AUC, AP, or classification report as a way to choose C.

Keep these distinctions explicit:

- Removing constant or identical columns is structural cleanup, not evidence of a predictor's relationship with bankruptcy.
- L1 selection is supervised: it uses the target labels in its training data.
- A diagnostic variable list is not a proven best subset.
- A variable retained by the selector need not be used by the future tree.
- A zero coefficient does not prove that a variable is useless or unrelated to bankruptcy.
- A nonzero coefficient does not establish causation or statistical significance.
- A linear selector can miss information useful only through nonlinear relationships or interactions. The future tree without supervised selection is therefore an essential comparator.

Do not add forward selection, ridge fits, elastic net, PCA, SMOTE, class weighting, automatic correlation-based deletion, feature engineering, outlier clipping, or a second selector. They would change or broaden the agreed experiment.

## 5. Sources and course connections

Read the supplied project instructions and the relevant lecture pages. Cite the course in notebook Markdown using a short lecture title and PDF page number. Page references here are physical PDF page positions starting at 1, not necessarily the printed slide numbers.

| Source | Pages to use | Connection |
| --- | --- | --- |
| Gaussian regression and variable selection | 13, 18, 20–21 | Sparsity; ambiguity among correlated predictors; forward selection as background; training-only scaling |
| Regularization and variable selection | 2, 7–9, 16–20, 24 | L1 penalties and zero coefficients; tuning; support-recovery limits; descriptive selection stability; scope of guarantees |
| Statistical learning: empirical risk, generalization and minimax limits | 2–3, 15–16, 24 | Separation of fitting, selection and final evaluation; logistic loss in Appendix F |
| Decision trees using weak learners | 8 | Variables can matter through interactions |
| ProjectRequirements.pdf | 1–2 | Combined approach and fitting preprocessing/selection within each training split |

Be precise about what is an adaptation: the regularization lecture develops squared-error lasso in a Gaussian regression setting. We retain its L1 selection principle and use logistic loss because the outcome is binary. The classification slides introduce logistic loss, but do not present this entire combined procedure.

Do not claim that Gaussian minimax rates, orthogonal soft-thresholding formulas, or exact-support guarantees automatically apply to this dataset or to the future selection-plus-tree procedure.

Use these official external references sparingly in the notebook:

- Dataset: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- L1 feature selection: https://scikit-learn.org/stable/modules/feature_selection.html#l1-based-feature-selection
- Logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Leakage prevention: https://scikit-learn.org/stable/common_pitfalls.html

The project requires full, verbatim disclosure of relevant LLM exchanges in the final report. Mention this briefly in the notebook's scope/reproducibility section. Do not fabricate a transcript or claim this notebook itself satisfies that appendix requirement.

## 6. Fixed configuration

Put the following settings in one early Python cell. Use them consistently throughout the notebook and record them in the manifest.

| Setting | Value |
| --- | --- |
| Data path | `data/data.csv` |
| Output directory | `outputs/variable_selection` |
| Target column | `Bankrupt?` |
| Split and fold seed | 42 |
| Test proportion | 0.20 |
| Cross-validation folds | 5, stratified, shuffled |
| Candidate C values | `[0.01, 0.1, 1.0, 10.0]` |
| Scaling | `StandardScaler`, fitted on each training subset |
| Logistic penalty | Pure L1 |
| Solver | `saga` |
| Intercept | Included and unpenalized |
| Class weighting | `None` |
| Estimator random state | 42 |
| Initial maximum iterations | 20000 |
| Solver tolerance | `1e-4` |
| One permitted convergence retry | Increase maximum iterations to 100000 |
| Selection threshold | Retain a predictor if `abs(coefficient) >= 1e-6` |

Check the installed scikit-learn version and use its supported pure-L1 syntax. Current releases use `l1_ratio=1`; older releases use `penalty="l1"`. Use one appropriate constructor for the environment and explain this briefly if necessary. Do not build a large version-compatibility framework.

Explain that C controls inverse regularization strength: smaller C means stronger penalization. Do not equate C to the lecture's lambda. Their numerical relationship depends on the objective normalization.

Do not tune the numerical coefficient threshold, solver tolerance, random seed, or the candidate grid based on attractive-looking variable lists.

## 7. Notebook section 1: purpose and what will remain undecided

Start with a short Markdown introduction explaining:

1. There are many financial indicators, some redundant or potentially unhelpful.
2. We will use L1 logistic regression to create several candidate input sets.
3. We will examine how the sets change with regularization and with the training sample.
4. We will preserve a test set and reproducible development folds for the following stage.
5. This notebook will not choose the final C, train a tree, or report final predictive performance.

Make the positive class explicit. Describe rows as company-related financial records, not necessarily unique companies. The supplied file lacks explicit company identifiers and observation dates; it does not establish a particular forecasting horizon or predictor timing. Keep the exercise framed as retrospective analysis of recorded bankruptcy labels.

Follow with the imports/configuration Python cell. Use standard-library modules plus NumPy, pandas, matplotlib, and scikit-learn. Jupyter execution support is an environment requirement; no additional visualization or modeling library is necessary.

## 8. Notebook section 2: load and check the data

Explain the distinction between a row, a predictor column, and the target column. Load the CSV, strip surrounding whitespace from column names, and assert that cleaned names remain unique.

Create a stable `source_row_id` equal to the zero-based row position in the original CSV. Keep it as an index or separate bookkeeping object. Never include it among the predictors. Do not reorder the original input before assigning it.

Perform basic integrity checks and display one small audit table:

- Number of rows and columns.
- Number of candidate predictor columns.
- Target values and class counts.
- Whether all predictors are numeric.
- Counts of missing values and nonfinite values.
- Counts of exact duplicate full rows and duplicate predictor rows.

The previously inspected uploaded version has 6,819 rows, 95 predictors, 220 bankrupt records, no missing/infinite values, and no duplicate rows. Recompute these figures; do not paste them as measured outputs. Treat a materially different file as something to investigate, not as permission to force the expected numbers.

Compute a SHA-256 hash of the original CSV bytes using `hashlib`. The future notebook will use it to check that saved row assignments still refer to the same file and row order.

Do not impute values, drop records, or guess numeric corrections. If missing/nonfinite values or duplicate records are found, stop with an informative explanation so the data issue can be resolved explicitly. For the inspected file, no such repair is needed.

Include a compact class-count plot with counts and percentages. Explain that bankrupt records are uncommon, making variable stability and later validation especially important. Do not add a classifier baseline here.

## 9. Notebook section 3: establish the development/test boundary

Explain these roles before the code:

- Development data are available for learning the procedure and later choosing its settings.
- Test data are reserved for the final assessment after all choices are frozen.
- Stratification preserves approximately the same class proportions in both sets.

Split the source row positions with `train_test_split`, `test_size=0.20`, `stratify=y`, and `random_state=42`.

Immediately sort the resulting development row IDs and test row IDs in ascending source-row order. Then construct the development and test datasets from those IDs. Sorting after splitting does not change membership and makes later reconstruction unambiguous.

Print a small count table. For the inspected file, the expected allocation is 5,455 development records with 176 bankruptcies and 1,364 test records with 44 bankruptcies. Assert disjoint membership and complete coverage of all source row IDs.

After splitting, do not use test predictors for descriptive statistics, scaling, cleanup, or selection. Do not use test labels beyond the split/count integrity checks. Do not export a selected test-feature matrix.

Explain that random stratification does not establish independence between firms or reliability across time. The missing identifiers and dates prevent checking those aspects.

## 10. Notebook section 4: create and save the five development folds

Explain cross-validation with a concrete description: divide development data into five groups; train on four and reserve the fifth; repeat so each record is held out once.

Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on development rows ordered as above. Materialize the folds once as a list of training/validation position pairs relative to that development dataframe.

Map fold positions back to `source_row_id`. Assign every development row exactly one validation-fold label from 1 to 5. Test rows have no fold label. Assert that validation groups are disjoint and together cover development data exactly once.

Display fold training/validation sizes and class counts. Each validation fold is expected to contain approximately 35 bankruptcies.

This notebook fits selectors only on each fold's training portion. Its validation labels are not used to score or tune the selector. These folds serve two purposes: descriptive checks of how selections vary across training samples, and a reusable evaluation structure for the future tree notebook.

Save `split_assignments.csv` with exactly these columns:

| Column | Meaning |
| --- | --- |
| `source_row_id` | Original zero-based CSV row position |
| `split` | `development` or `test` |
| `cv_validation_fold` | 1–5 for development rows; blank for test rows |

Write rows in ascending `source_row_id` order and do not export an extra dataframe index. The CSV hash in the manifest must accompany this file.

## 11. Notebook section 5: structural cleanup and scaling

Explain why a column with the same value in every training record cannot distinguish outcomes, and why two identical columns contain the same recorded information.

Use a simple training-only cleanup function:

1. Identify columns constant within the given training subset.
2. Among the remaining columns, compare exact equality of the full training-column values.
3. For every identical group, keep the first column in original CSV order and mark later columns as duplicates of that retained representative.
4. Return the retained names and a table of reasons for every original predictor.

Use exact equality, not rounded values or `allclose`: values in this dataset have very different scales. Do not treat merely high correlation as duplication. The fitted list of retained names is what would be applied to validation rows; never relearn the list on validation data.

The inspected file contains the constant `Net Income Flag` and these duplicate pairs:

- `Current Liabilities/Liability` and `Current Liability to Liability`.
- `Current Liabilities/Equity` and `Current Liability to Equity`.

Treat these as observations to verify, not a hard-coded universal deletion list. Fold-specific constant columns may differ.

Explain standardization with:

\[
z_{ij}=\frac{x_{ij}-\mu_j}{s_j},
\]

where x is a recorded value, mu is its training-column mean, and s is its training-column standard deviation. Use `StandardScaler` rather than writing a manual implementation with a different standard-deviation convention.

The point is to put predictors on comparable scales for coefficient penalization. Standardization does not change a predictor into a different economic concept, repair faulty encoding, make it normally distributed, or remove the influence of extreme observations.

Illustrate the scale issue with a small development-only summary of a few predictors. The inspected data include values up to 10 billion, so do not say all predictors are normalized to [0, 1]. Retain the recorded values in the main procedure; do not add clipping or guessed corrections.

The future tree should receive the retained columns in their original supplied scale. Standardization belongs to the selector. Original supplied values may already be transformed, so do not label their thresholds as real financial units without documentation.

## 12. Notebook section 6: explain L1 logistic selection

Use two or three short Markdown cells, separated by small illustrative Python cells only where those calculations help understanding. Do not add code solely to force an illustration.

Explain the components in this order:

1. A coefficient is the weight the fitted linear model gives to a standardized predictor.
2. Logistic regression turns a weighted sum into a number between 0 and 1 representing an estimated probability of bankruptcy.
3. Logistic loss penalizes probabilities inconsistent with observed outcomes; confidently wrong probabilities incur more loss.
4. Regularization adds a cost for coefficient magnitude.
5. The L1 penalty uses absolute coefficient values and can make coefficients exactly zero.
6. Variables with retained nonzero coefficients become candidate inputs to the tree.

Show the equations with every symbol explained:

\[
q_i=\frac{1}{1+\exp[-(b_0+z_i^\top\beta)]},
\]

\[
\min_{b_0,\beta}\;\frac1n\sum_i[-y_i\log q_i-(1-y_i)\log(1-q_i)]
+\lambda\sum_j|\beta_j|.
\]

Explain n, y, q, z, beta, the intercept b0, and the penalty strength lambda. The intercept sets the baseline level and is not included in the L1 penalty here.

A short hypothetical coefficient example is sufficient: coefficients `(0.6, 0, -0.2)` retain the first and third predictors. Label it hypothetical. Do not present the course's orthogonal soft-thresholding formula as the algorithm for correlated logistic regression.

Explain that stronger regularization generally encourages a smaller subset, but selected variables and subset sizes need not change monotonically at every step for correlated data and numerical fits. Do not enforce monotonicity in code.

## 13. Notebook section 7: define the small fitting routine

Write one short function that fits selection on an explicitly supplied training dataframe and training target. It should call the cleanup helper, fit `StandardScaler`, fit the configured logistic estimator, and return named results. A dictionary is sufficient; no custom estimator class is needed.

Return enough information to construct the exported diagnostics:

- Original predictor names and structural-cleanup statuses.
- Retained names after cleanup and duplicate representatives where applicable.
- Fitted scaler and logistic model, for inspection in memory.
- Standardized coefficients mapped back to the retained names.
- Selected names using the fixed absolute-coefficient threshold.
- Number of iterations, convergence status, and the iteration limit used.

Do not pass validation or test data to this function. Do not inspect their values from a global variable inside the function. The fit inputs must make its information boundary obvious.

Handle convergence explicitly. Capture `ConvergenceWarning`, rerun once from a fresh estimator with the larger iteration limit if necessary, and record the result. Do not suppress a warning and call the fit successful. Other fitting errors should raise clear errors rather than being swallowed.

If a fit still fails to converge, record that candidate/fold as invalid and do not treat its coefficients as valid selection evidence. Keep the failure visible in summaries. Do not average only the successful folds and make an invalid candidate look comparable.

If a converged fit selects no variables, record that result without adding a predictor back. It is informative about excessive regularization, but that candidate cannot supply a usable input set for the future tree on that training fold.

## 14. Notebook section 8: fit candidates within the training folds

Run a readable nested loop over the five saved folds and four C values: 20 selector fits, plus any permitted convergence retries.

For each fit:

- Supply only that fold's training rows and their labels.
- Refit structural cleanup, scaling, and coefficients from those rows.
- Record the training sample size and positive count.
- Record structural exclusions, coefficients, and selected predictors.
- Do not compute validation classification scores or pick a winning C.

Cleanup and scaling could be reused across C values within a fold, but performance optimization is not required. Prefer the clearer implementation. Nothing fitted may be reused across different training folds.

Display a compact fit summary and explain that each setting has been examined on five slightly different training samples. A selector fitted on all development labels would not be a valid substitute for these fold-specific fits in a later cross-validation experiment.

## 15. Notebook section 9: describe subset size and stability

Produce one small table, with C values in ascending numerical order, showing:

- Number of converged folds out of five.
- Number of folds selecting zero variables.
- Mean, minimum, and maximum selected-variable counts when all five fits converged.
- Whether the candidate has five converged, nonempty fold selections and is therefore numerically eligible for the later tree search.

If any fold is nonconverged, leave the aggregate selection-count statistics unavailable and explain why. Numerical eligibility is not a performance recommendation. Do not label any row best, optimal, recommended, or the winner.

Make one labeled plot of mean selected-variable count against C on a logarithmic x-axis, showing the fold range. Plot only candidates with complete converged fold results; identify invalid candidates separately. Explain the direction of regularization on the axis.

For each C with five converged fits, count how many of the five training-fold selectors retained each original predictor. A structurally excluded predictor counts as not selected in that fold. Display a compact table of the most frequently retained variables, with a fixed display rule such as the top 15 ordered by selection count and then original CSV order. The display limit does not change selection.

Save complete results even when the displayed table is abbreviated. Do not create a consensus subset by taking a union, intersection, or arbitrary frequency threshold. That would introduce a new selection method.

Explain that the training folds overlap, so these counts are descriptive. They are not independent replications, confidence intervals, p-values, or probabilities that a predictor is truly relevant. The course's bootstrap stability exercise is related but different; do not claim that five overlapping folds implement that exercise.

Do not add a second resampling study to the main notebook. The five-fold diagnostic is enough for this focused contribution.

## 16. Notebook section 10: fit four full-development candidate subsets

Explain why these additional fits are useful: they show what each C retains when all currently available development observations are used. They are diagnostic candidate subsets, not final chosen subsets and not valid fixed inputs for later cross-validation.

Fit the same procedure on all development rows once for each C: four additional fits. Keep the test set untouched. Use the same convergence and zero-feature rules.

Present the full-development selected count beside each C and a readable selected-variable table with standardized coefficients. If a table is long, show a limited preview and point to the complete exported table. Do not manually shorten the actual selected list.

Explain coefficient signs as associations within this fitted logistic selector, conditional on its other included predictors. Do not interpret magnitude as causal importance, attach significance stars, or claim that it describes a future tree's effects. With correlated indicators, coefficients and selected names can change while predictions remain similar.

State prominently:

> These subsets use all development labels. The next notebook must refit selection inside each training fold when comparing trees. It must not cross-validate a tree using one of these full-development lists as a fixed input list.

## 17. Notebook section 11: export a reproducible handoff

Use CSV and JSON, not pickle or a collection of opaque fitted-model files. The next stage should refit the transparent procedure from raw data and saved split assignments. Do not export transformed copies of the dataset.

Write the five files listed in Section 2. They should contain these fields:

### `run_manifest.json`

- A small schema version, for example 1.
- Source data path, raw-file SHA-256 hash, row count, and ordered cleaned predictor names.
- Target name, label meanings, and zero-based source-row-ID convention.
- Development/test split settings and the post-split ascending-row-order convention.
- Cross-validation settings and fold labels 1–5.
- All selector/scaler parameters, C values, coefficient threshold, and convergence retry policy.
- Python, NumPy, pandas, and scikit-learn versions.
- Names of the other output files and their meanings.
- `final_C: null` and an explicit status that C selection is deferred to downstream tree validation.

### `split_assignments.csv`

Use the exact three-column specification in Section 10. Never store the target or predictor values here.

### `selection_fits.csv`

One row per attempted scope/fold/C combination, recording its final retry outcome rather than duplicating it as a second independent fit:

`fit_scope`, `fold`, `C`, `n_training_rows`, `n_training_positive`, `n_predictors_after_cleanup`, `n_selected`, `converged`, `n_iter`, `max_iter_used`, `usable_nonempty_selection`.

Use `cv_training` and `full_development` for scope. Fold is 1–5 for cross-validation fits and blank for full-development fits. There should be 24 rows if all planned fits are attempted. For nonconverged fits, leave `n_selected` unavailable and set usability to false. For converged empty selections, record zero and usability false.

### `feature_selection.csv`

One row per original predictor per scope/fold/C combination:

`fit_scope`, `fold`, `C`, `feature`, `cleanup_status`, `duplicate_of`, `standardized_coefficient`, `selected`.

Use cleanup statuses `retained`, `constant`, and `duplicate`. Coefficients are unavailable for structurally excluded predictors; their selected flag is false. For nonconverged fits, coefficients and selected flags are unavailable, so missing results cannot masquerade as stable exclusion. Keep empty fields distinct from actual numeric zeros.

### `candidate_summary.csv`

One row per C:

`C`, `n_converged_folds`, `n_zero_feature_folds`, `cv_mean_selected`, `cv_min_selected`, `cv_max_selected`, `cv_numerically_eligible`, `full_development_converged`, `full_development_n_selected`.

This is a diagnostic summary, not a performance ranking. Apply the missing-result conventions from Section 15.

Write deterministic row ordering and `index=False` for CSV outputs. Use Python-native types when serializing JSON. Repeat runs may update these five generated outputs, but must not alter input files or unrelated outputs.

## 18. Notebook section 12: checks and honest findings

Before the final handoff Markdown cell, include a compact verification Python cell preceded by a plain-language explanation. Use meaningful assertions rather than a separate testing framework:

- Source-row IDs are unique and cover the complete source file.
- Development and test membership are disjoint.
- Each development row belongs to exactly one validation fold and no test row has a fold label.
- All training folds contain both outcome classes.
- The target and source-row ID never appear in predictor lists.
- For converged fits, selected names match the fixed coefficient-threshold rule.
- Recorded selected counts agree with the feature-level export.
- Converged and invalid fits use the documented result conventions.
- Every planned scope/fold/C combination appears exactly once in the fit summary.
- The five exported files exist and can be read back.
- Reconstructed source-row membership from the saved assignments matches the in-memory split.
- The manifest records `final_C` as null.

Run the completed notebook from top to bottom in a fresh kernel using the project's environment. Resolve genuine execution failures, broken paths, and misleading labels. Do not create a fake result to make an assertion pass. If the environment cannot execute it or a numerical failure remains, preserve the notebook and state the exact limitation.

Write a short results-based Markdown interpretation before the final handoff. Summarize how subset sizes differ across C, whether selections are consistent across folds, and any invalid settings or convergence difficulties. Avoid predictive claims because no downstream classifier has been evaluated.

The final response to the user should identify the notebook and exported diagnostics, say whether a clean execution succeeded, and state that no final C was chosen. Do not claim that preparing the selection notebook completes the full project.

## 19. Required final notebook Markdown cell: how the next stage chooses C

The notebook must end with a substantial, beginner-friendly Markdown explanation covering all of the following, in this order. It must stand on its own for the teammate building the decision-tree notebook. Explain the procedure without implementing it here.

### What we have learned, and what we have not

We now know what each candidate regularization setting tends to retain and how sensitive those selections are to the training sample. We do not yet know which setting gives the best inputs to a decision tree. A smaller subset is not automatically better, and a more stable subset is not automatically more predictive.

C influences the information available to the tree. Consequently, its final choice must be based on validation of the complete process: training-only cleanup, training-only scaling, L1 selection, tree fitting, and conversion of the tree's score into an alert.

### Step 1: recover exactly the same data and splits

The next notebook must load the raw CSV, apply the same whitespace cleanup, and verify its raw-file SHA-256 hash against `run_manifest.json`. It must reconstruct development/test membership and validation folds from `split_assignments.csv`, ordering rows by `source_row_id`. It must not create a new favorable split or assume that the same seed is sufficient when row ordering might have changed.

The test data remain unused during every model and threshold comparison below.

### Step 2: fit the complete procedure inside every saved training fold

For each saved fold and each C:

1. Use only the fold's training records to identify constant/duplicate columns.
2. Fit the scaler and L1 logistic model on those training records and labels.
3. Determine the selected names using the same coefficient threshold.
4. Train candidate decision trees on the selected training columns in their original supplied scale.
5. Apply the already learned column selections to the held-out validation records.
6. Obtain the tree's validation bankruptcy scores and evaluate the alert rules described below.

The diagnostic full-development subsets from this notebook must not be substituted at step 3. They used labels from the held-out fold and would leak information into the comparison. Likewise, selecting a consensus set across the five diagnostic folds and treating it as fixed would introduce an additional data-dependent selection step.

A different variable list in each fold is normal. We are comparing a procedure with a fixed C, not requiring the same selected names in every training sample.

### Step 3: compare a small, predeclared grid

The agreed next-stage tree is an unweighted `DecisionTreeClassifier` using Gini impurity, `random_state=42`, and no additional pruning search. The planned settings are:

| Component | Candidate values |
| --- | --- |
| L1 selector C | 0.01, 0.1, 1, 10 |
| Tree maximum depth | 2, 3, 4 |
| Minimum observations per leaf | 20, 50 |
| Alert threshold on class-1 tree score | 0.02, 0.05, 0.10, 0.20, 0.50 |

Explain maximum depth as the number of successive questions along a tree path, and minimum leaf size as the minimum training evidence required for a terminal group.

The tree supplies a bankruptcy score for each record. An alert is issued when that score is greater than or equal to the threshold. The threshold affects how many bankruptcies are detected and how many nonbankrupt records receive false alerts. Changing the threshold requires rescoring predictions, not refitting the tree.

Settings with unresolved selector convergence failures or no selected variables in any fold are invalid for this combined comparison. Record them transparently; do not average a candidate over fewer than five valid folds, add a variable by hand, or conceal failures.

### Step 4: use one declared primary metric

Select the complete configuration using mean validation balanced accuracy across the five folds:

\[
\mathrm{Balanced\ accuracy}
=\frac12\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right).
\]

Explain TP as correctly alerted bankrupt records, FN as missed bankrupt records, TN as correctly unflagged nonbankrupt records, and FP as false alerts. The first fraction is the detection rate among bankrupt records; the second is the correct non-alert rate among nonbankrupt records. Their average gives equal importance to the two classes despite their unequal sizes.

Ordinary accuracy alone is unsuitable for choosing this screening rule because predicting nonbankruptcy for everyone would be correct for roughly 97% of this dataset while detecting no bankruptcies. Balanced accuracy represents a declared statistical preference; it does not establish real financial error costs.

Recall, precision, average precision, alert counts, subset size, and stability remain useful supporting evidence. Do not switch the winning criterion after inspecting which metric favors a preferred subset.

### Step 5: select C jointly with the tree and threshold

For each valid `(C, depth, minimum leaf size, threshold)` combination, calculate its five fold scores and their mean. Choose the combination with the highest mean balanced accuracy. The C belonging to that combination is the chosen C for the combined procedure.

Use the already agreed deterministic tie rules: treat differences below `1e-12` as numerical ties, then prefer fewer mean selected predictors, lower maximum depth, larger minimum leaf size, smaller C, and finally a higher alert threshold.

Do not pick C from the logistic model's own predictive performance, the smallest variable count, the most attractive financial story, or this notebook's stability table. Those criteria answer different questions.

The search consumes validation information. Its winning score is a development result, not an unbiased final assessment, and the five fold scores are dependent because training sets overlap.

### Step 6: establish whether selection adds value

The next stage must also evaluate a shallow tree using all predictors remaining after training-only structural cleanup, with the same saved folds, depth/leaf-size grid, thresholds, and primary metric. Include a constant majority-class baseline.

This comparison asks whether supervised selection actually improves or simplifies the tree usefully. It may reveal that the all-predictor tree performs as well or better. That would be a legitimate result, not a reason to conceal the comparison or tune against the test set.

### Step 7: freeze, refit, and assess once

After C, tree settings, and the alert threshold are chosen, freeze them. Refit the complete selected procedure on all development observations: cleanup, scaling, L1 selection, and then the tree. The resulting full-development selected names become the final input subset for that fitted procedure. Record separately how many of those inputs the tree actually uses in its splits.

Only then evaluate the frozen procedure on the untouched test set. Do not change C, features, depth, leaf size, or threshold after seeing its results. A disappointing test result is evidence to discuss, not permission to reuse the same test set for another selection round.

### What the next stage must return to this contribution

The variable-selection work becomes complete when the tree-validation stage returns:

- The chosen C together with the chosen tree settings and alert threshold.
- The five validation scores, their mean, and the predeclared comparison that justified the choice.
- A comparison against the all-predictor tree and the constant baseline.
- The final development-fitted selected-variable list and coefficients.
- The distinction between selected inputs and variables actually used by the tree.
- The final untouched-test assessment of the frozen procedure, with appropriate uncertainty and limitations.

Until that evidence exists, the correct conclusion is that this notebook has produced reproducible candidate selections and diagnostics, and that the final choice of C remains open.
