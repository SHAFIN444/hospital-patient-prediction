# Hospital Patient Prediction

Predicting a diabetes diagnosis from eight routine health measurements, using a decision tree.

A machine learning coursework project. The notebook cleans the data, trains a
`DecisionTreeClassifier`, and evaluates it on held-out patients. It reaches **79% accuracy**,
but it misses about **28% of actual diabetics** — which matters more than the headline number,
and is discussed below.

## The problem

Can we predict whether a patient has diabetes (yes / no) from measurements that are already
in a routine patient file?

If so, the value is in triage rather than diagnosis: flagging higher-risk patients early so a
doctor can order confirmatory tests sooner, turning data a clinic already holds into a fast,
low-cost screening signal.

## The dataset

The Pima Indians Diabetes Database: 768 patients, 8 features, 1 binary target (`Outcome`).
About 35% of the patients in the dataset are diabetic.

| Feature | Meaning |
| --- | --- |
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Blood sugar level |
| `BloodPressure` | Diastolic pressure |
| `SkinThickness` | Skin fold thickness |
| `Insulin` | Serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Family-history score |
| `Age` | Patient age |

Source: UCI Machine Learning Repository, obtained via
[Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Method

1. **Clean.** Several columns use `0` where a value is actually missing — a blood pressure or
   BMI of zero is not physically possible. These were converted to missing values and filled
   with the column median. This step mattered more than anything else in the project (see
   Results).
2. **Split.** 80% of the patients for training, 20% (154 patients) held back for testing.
3. **Train.** `DecisionTreeClassifier` with `max_depth=4`. Limiting the depth forces the tree
   to learn general rules instead of memorising individual patients.
4. **Evaluate.** Predict on the held-out patients and inspect the confusion matrix.

## Results

**Accuracy: 79%** on 154 held-out patients.

Confusion matrix:

| | Predicted: No | Predicted: Yes |
| --- | --- | --- |
| **Actual: No** | 83 | 17 |
| **Actual: Yes** | 15 | 39 |

### What the number hides

Accuracy alone is misleading here. Of the 54 patients in the test set who actually had
diabetes, the model caught 39 and **missed 15** — a recall of about 72%, meaning roughly
one in four diabetic patients is told they are fine.

In a screening context, those 15 false negatives are the dangerous errors. A false positive
costs a confirmatory test; a false negative may delay treatment. Any real deployment would
want to trade some accuracy for higher recall, for example by adjusting the decision
threshold or weighting the classes.

### What cleaning the data was worth

An earlier version of the model, trained before the impossible zeros were handled and without
a depth limit, scored **68%** and was clearly overfitting. Treating the hidden missing values
and capping the tree depth at 4 took it to **79%** on unseen patients.

The modelling choice was not the interesting part of this project; the data cleaning was.

### What the tree learned

The first and strongest split is on `Glucose`, which accounts for roughly 70% of the model's
feature importance. That is a sensible result — it agrees with how diabetes is actually
screened — and it is a useful sanity check that the model learned something real rather than
an artefact of the data.

## Ethical considerations

- **Population bias.** Every patient in this dataset is a Pima woman. The model should not be
  applied to other populations without retraining and revalidation, and its error rates on any
  other group are unknown.
- **This is not a diagnosis.** It is a screening aid at best. A clinician has to stay in the
  loop, and the false-negative rate above is the reason why.
- **Privacy and consent.** Patient health data requires strict handling, access control, and
  informed consent. This project uses a public, de-identified research dataset; real clinical
  data would not be usable this way.

## Limitations

- Trained and tested on 768 patients from a single population; the test set is only 154 people,
  so the accuracy figure carries meaningful uncertainty.
- Missing values were filled with the column median, which is a crude imputation and distorts
  the distribution of those columns.
- A single decision tree was used, with no cross-validation, no hyperparameter search, and no
  comparison against other models or a trivial baseline.
- Optimised for accuracy, not for recall, which is the wrong objective for a screening tool.

## Running it

The notebook is `diabetes_prediction.ipynb`. It runs in Google Colab or any local Jupyter
install with `pandas`, `scikit-learn`, and `matplotlib`.

```bash
pip install pandas scikit-learn matplotlib
jupyter notebook diabetes_prediction.ipynb
```
