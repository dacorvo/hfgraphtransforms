# Evaluating the effect of graph transformations on a BeRT model

## Setup

````
python -m venv hg
source ./hg/bin/activate
pip install -r requirements.txt
````

## Usage

````
usage: bert-evaluation.py [-h] [--device DEVICE] [--infer] [--evaluate]
                          [--transform]

HuggingFace Bert QuestionAnswering tester

optional arguments:
  -h, --help       show this help message and exit
  --device DEVICE  Target device.
  --infer          Pass a simple question to the model.
  --evaluate       Evaluate the model score on the squad2 validation dataset.
  --transform      Test several model transformations.
````

Three transformations from optimum.fx.optimizations.transformation that are relevant
for a Transformer model are selected:
- MergeLinears,
- FuseBiasInLinear,
- ChangeTrueDivToMulByInverse.

Note: I could not apply the FuseBiasInLinear transform on the model due to a shape mismatch.
I think it happens in the score x value matmul, which is odd because my understanding is that Linear output
shapes should not be impacted (only inputs and weights).
I quickly checked the corresponding unit test in optimum and it is supposed to work out-of-the-box for Bert.

An additional Transformation to remove Dropout layers is also added.

## Evaluation on a simple question

### Apple M1 Max - pytorch 1.31.1 - CPU

````
Base average inference: 597.0487041999999 ms
Base+LintAndRecompile average inference: 593.5082249999999 ms
Base+MergeLinears average inference: 596.3989666 ms
Base+ChangeTrueDivToMulByInverse average inference: 593.9272542000001 ms
Base+RemoveDropout average inference: 591.5362708 ms
Base+ComposeTransformation average inference: 595.0795458999997 ms
````

### Apple M1 Max - pytorch 1.13.1 - Metal Performance Shaders (MPS)

````
Base average inference: 93.70445829999996 ms
Base+LintAndRecompile average inference: 93.08724579999996 ms
Base+MergeLinears average inference: 92.29938749999995 ms
Base+ChangeTrueDivToMulByInverse average inference: 93.05135 ms
Base+RemoveDropout average inference: 91.86703329999997 ms
Base+ComposeTransformation average inference: 83.90547500000008 ms
````

### Apple M1 Max - pytorch 2.1.0a0+gitec3894e - CPU

````
Base average inference: 107.75942089999998 ms
Base+LintAndRecompile average inference: 107.88847500000003 ms
Base+MergeLinears average inference: 96.59027919999997 ms
Base+ChangeTrueDivToMulByInverse average inference: 107.4367917 ms
Base+RemoveDropout average inference: 106.66231249999996 ms
Base+ComposeTransformation average inference: 96.183875 ms
````

### Apple M1 Max - pytorch 2.1.0a0+gitec3894e + SEELF - CPU

````
Base average inference: 104.6668875 ms
Base+LintAndRecompile average inference: 103.52200420000005 ms
Base+MergeLinears average inference: 90.44809999999997 ms
Base+ChangeTrueDivToMulByInverse average inference: 102.8573583 ms
Base+RemoveDropout average inference: 103.28010000000009 ms
Base+ComposeTransformation average inference: 89.82156670000005 ms
````

### Apple M1 Max - pytorch 2.1.0a0+gitec3894e + SEELF - Metal Performance Shaders (MPS)

````
Base average inference: 31.323462500000065 ms
Base+LintAndRecompile average inference: 31.18443339999999 ms
Base+MergeLinears average inference: 28.317558300000023 ms
Base+ChangeTrueDivToMulByInverse average inference: 32.83145830000001 ms
Base+RemoveDropout average inference: 29.854908300000016 ms
Base+ComposeTransformation average inference: 27.32291250000003 ms
````

## Evaluation on a subset of a squad dataset

### Apple M1 Max - pytorch 1.13.1 - CPU - 10 samples

````
Base : f1 = 30.00, samples/s = 0.3959, latency = 2525.9091 ms
Base+LintAndRecompile : f1 = 30.00, samples/s = 0.3963, latency = 2523.5268 ms
Base+MergeLinears : f1 = 30.00, samples/s = 0.3949, latency = 2532.4383 ms
Base+ChangeTrueDivToMulByInverse : f1 = 30.00, samples/s = 0.3959, latency = 2525.6471 ms
Base+RemoveDropout : f1 = 30.00, samples/s = 0.3954, latency = 2528.7776 ms
Base+ComposeTransformation : f1 = 30.00, samples/s = 0.3944, latency = 2535.2992 ms
````

### Apple M1 Max - pytorch 1.13.1 - Metal Performance Shaders (MPS) - 100 samples

````
Base : f1 = 43.89, samples/s = 10.0389, latency = 99.6123 ms
Base+LintAndRecompile : f1 = 43.89, samples/s = 10.1523, latency = 98.5000 ms
Base+MergeLinears : f1 = 43.89, samples/s = 7.4352, latency = 134.4946 ms
Base+ChangeTrueDivToMulByInverse : f1 = 43.89, samples/s = 9.2752, latency = 107.8143 ms
Base+RemoveDropout : f1 = 43.89, samples/s = 10.1788, latency = 98.2434 ms
Base+ComposeTransformation : f1 = 43.89, samples/s = 10.8458, latency = 92.2019 ms
````

### Apple M1 Max - pytorch 1.13.1 - Metal Performance Shaders (MPS) - 1000 samples

````
Base : f1 = 45.59, samples/s = 10.2672, latency = 97.3979 ms
Base+LintAndRecompile : f1 = 45.59, samples/s = 10.3077, latency = 97.0146 ms
Base+MergeLinears : f1 = 45.59, samples/s = 8.6649, latency = 115.4085 ms
Base+ChangeTrueDivToMulByInverse : f1 = 45.59, samples/s = 9.7583, latency = 102.4771 ms
Base+RemoveDropout : f1 = 45.59, samples/s = 10.1991, latency = 98.0478 ms
Base+ComposeTransformation : f1 = 45.59, samples/s = 10.8363, latency = 92.2825 ms
````

### Apple M1 Max - pytorch 2.1.0a0+gitec3894e + SEELF - CPU

````
Base : f1 = 30.00, samples/s = 5.2294, latency = 191.2265 ms
Base+LintAndRecompile : f1 = 30.00, samples/s = 5.2172, latency = 191.6719 ms
Base+MergeLinears : f1 = 30.00, samples/s = 5.5387, latency = 180.5486 ms
Base+ChangeTrueDivToMulByInverse : f1 = 30.00, samples/s = 5.2830, latency = 189.2865 ms
Base+RemoveDropout : f1 = 30.00, samples/s = 5.2820, latency = 189.3220 ms
Base+ComposeTransformation : f1 = 30.00, samples/s = 5.5599, latency = 179.8609 ms
````

### Apple M1 Max - pytorch 2.1.0a0+gitec3894e + SEELF - Metal Performance Shaders (MPS) - 100 samples

````
Base : f1 = 43.89, samples/s = 22.2120, latency = 45.0206 ms
Base+LintAndRecompile : f1 = 43.89, samples/s = 22.1101, latency = 45.2283 ms
Base+MergeLinears : f1 = 43.89, samples/s = 19.9722, latency = 50.0696 ms
Base+ChangeTrueDivToMulByInverse : f1 = 43.89, samples/s = 19.7460, latency = 50.6432 ms
Base+RemoveDropout : f1 = 43.89, samples/s = 22.1085, latency = 45.2315 ms
Base+ComposeTransformation : f1 = 43.89, samples/s = 22.2645, latency = 44.9146 ms
````



