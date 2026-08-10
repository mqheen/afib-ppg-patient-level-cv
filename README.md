# AFib detection in ICU data: what happens when you split by segment instead of by patient

I built this after finishing a course project where I'd claimed patient-level
data splitting was the right choice, and then realized I had never actually
checked what the wrong choice would have produced. This is that check.

Short version: on 6,017 ICU patients, splitting by segment instead of by patient
raises AUROC by about 0.13. And a model given nothing but age, sex, weight and
height, which cannot possibly know what rhythm someone is in at 3pm versus 4pm,
scores 0.956 under segment-level splitting. That number is the whole point of the
repo.

## Results

Same features, same model, same seed. The only thing I changed was whether the
train/test boundary falls between patients or between segments.

| Feature set | Patient-level | Segment-level |
|---|---|---|
| V1 aggregate (13 features) | 0.837 | 0.970 |
| V2 + irregularity (26 features) | 0.849 | 0.972 |
| Demographics only (4 features) | 0.644 | 0.956 |

AUROC is generous here because AF is only 9.4% of segments, so average precision
is the number I'd actually look at:

| Feature set | Patient-level | Segment-level |
|---|---|---|
| V1 aggregate | 0.420 | 0.822 |
| V2 + irregularity | 0.487 | 0.830 |
| Demographics only | 0.111 | 0.753 |

The demographics row is what convinced me. Age, sex, weight and height don't
change between one 30-second segment and the next, so there's no way for them to
track rhythm. Under patient-level splitting they get 0.644, about what you'd
expect given that AF really is more common in older patients. Under segment-level
splitting the same four numbers reach 0.956, which is basically what my full
model gets. The only thing they can be doing is identifying the patient and
looking up the answer.

In the segment-level runs, 5,584 of 6,017 patients had segments in both training
and test. The script prints that count on every run, which I added after
second-guessing whether my patient-level splitter was actually working.

## Background

This was my half of a two-person project for AI and Healthcare at Barnard,
evaluating AFib detection on MIMIC-III-Ext-PPG, which was released in February
2026 and is much larger than anything previously available for this problem
(6,131 patients, versus 37 in the paper we were building on).

Bulut et al. (2025) reported 95.17% accuracy on 37 patients using a segment-level
split. Kwon et al. (2019) ran both splitting strategies on the same data and
found segment-level inflated specificity, and that's the paper I cited at the
time to justify splitting by patient. So the finding isn't new. What I wanted was
to see it on my own data at a scale where the effect isn't arguable. The
demographics-only control is the part I haven't seen done elsewhere.

## What the features actually are

These come from the dataset's `metadata.csv`, not from the PPG waveform. In
MIMIC-III-Ext-PPG the heart rate values are computed from the concurrent ECG R-R
intervals and the blood pressure from the arterial line. So this is AFib
detection from per-segment vital sign summaries plus demographics, not
photoplethysmography classification, and I've tried not to call it that anywhere.

The leakage result doesn't depend on this and would hold for any feature set. But
it does mean the V2 features are window-level heart rate irregularity, not
beat-to-beat irregularity, which is what I was calling them for a while before I
went back and read my own code properly.

V1 is 13 features: median and IQR of heart rate, respiratory rate, systolic and
diastolic BP, plus the respiratory signal quality index and four demographics.

V2 adds 13 more, built from the three 10-second windows inside each 30-second
segment. The one I care about is `hr_irreg`, defined as |HR₁ − HR₀| + |HR₂ − HR₁|.
The idea is that AF produces an irregular ventricular response, so how much heart
rate moves *within* a segment should carry signal that a single median throws
away.

## Did V2 help

A bit. On patient-level evaluation average precision goes from 0.420 to 0.487 and
recall from 0.378 to 0.434. Real, not dramatic.

The thing I didn't expect: under segment-level splitting V1 and V2 are almost
identical, 0.970 against 0.972. Once the model can recognize patients it stops
needing the features, so the feature engineering looks worthless. If I had only
ever run the segment-level version I'd have concluded the irregularity features
did nothing, which is the opposite of what's true. Leakage doesn't just inflate
your score, it can hide whether your work mattered.

## Things wrong with this

- It isn't a usable detector. Average precision of 0.487 at 9.4% prevalence means
  most of what it flags is wrong. The point here is the evaluation, not the model.
- Missing blood pressure and respiratory values get zero-filled, so "no reading"
  and "a reading of zero" look the same to the model. This is bad and I know it's
  bad. Median imputation plus a missingness indicator is the fix and it's on the
  list below.
- One seeded subsample (1,000,000 of 6,399,754 segments, seed 42, covering 6,017
  of 6,189 patients), one split per arm, so no error bars.
- All from one ICU at one hospital. Nothing here says anything about whether these
  features transfer.
- Rhythm labels come from the chart within 15 minutes of the waveform rather than
  from reading the segment itself, so some of them are probably wrong.

One more, which I only found while building this: the original course notebook
read `metadata.csv` with `engine='python', on_bad_lines='skip'` and silently kept
265,578 of the 6.4 million rows. I didn't catch it until I rewrote the loader.
Everything in this repo uses the full file.

## Getting the data

MIMIC-III-Ext-PPG needs credentialed PhysioNet access and can't be
redistributed, so there's no data in this repo. You need to complete CITI "Data
or Specimens Only Research" training, apply for credentialed access, sign the
data use agreement, and download from
https://physionet.org/content/mimic-iii-ext-ppg/1.0.0/

Only `metadata.csv` is used. None of the waveform files are read, which means you
can skip most of the download.

## Running it

```bash
pip install -r requirements.txt
python src/build_features.py --data /path/to/mimic-iii-ext-ppg/1.1.0 --out features.npz
python src/leakage_experiment.py --features features.npz --subsample 1000000
```

`--subsample 0` uses all 6.4M segments and needs a lot more memory than free
Colab has. I crashed the runtime twice working that out. A million segments takes
around 20 minutes on Colab CPU.

Model is `HistGradientBoostingClassifier`, 300 iterations, depth 5, learning rate
0.1, balanced class weights. Threshold swept on the development set only, same
procedure in both arms.

## To do

- Median imputation with a missingness flag instead of zero-fill
- Repeat over several seeds so the inflation estimate has a range on it
- Compute real beat-to-beat intervals from the raw PPG with NeuroKit2, which
  would make this an actual PPG project rather than a vitals project

## Credit

Two-person course project with Kimberly Wei. Kimberly built the CNN arm, a
17-layer 1D convolutional network adapted from Bulut et al., which I benchmarked
my models against. Her code isn't in this repo. What's here is my side: the
features, the gradient-boosted models, and the leakage experiment, which I did
afterwards on my own.

## References

Bulut, M. G., Unal, S., Hammad, M., & Pławiak, P. (2025). Deep CNN-based
detection of cardiac rhythm disorders using PPG signals from wearable devices.
*PLOS ONE*, 20(2), e0314154.

Kwon, S., et al. (2019). Deep learning approaches to detect atrial fibrillation
using photoplethysmographic signals. *JMIR mHealth and uHealth*, 7(6), e12770.

Moulaeifard, M., Charlton, P. H., & Strodthoff, N. (2026). MIMIC-III waveform
database matched subset (PPG extension). *PhysioNet*.
