## 1. Project Overview

**Title:** A Comparative Study of Neural Models for Direction Aware NER and Relation Extraction in Transit Service Alerts

**Objective:**

The primary goal is to develop and compare machine learning models that can extract structured information from noisy MTA transit service alerts.

- **NER:** Identifying ROUTE  (e.g "M15") and DIRECTION (e.g "Northbound").
- **RE:** Linking each ROUTE  to its correct DIRECTION.

**Key Challenge:**

Transit alerts are short (Mean length: approximately 22 tokens) but they are dense, often containing multiple routes and directions (e.g *"Northbound M15 and Southbound M103 delays"*). The model must understand the scope and context.

## 2. Methodology & Design Choices

I implemented a **Weak Supervision** pipeline detailed below, i also referenced the specific code modules.

### 2.1. Data Preprocessing

Code: 1_data_preprocessing.py 

- **Source:** MTA Service Alerts (NYC Open Data).
- **My Design Choice:** Strict filtering to remove noise.
- **Agency Filter:** Kept only "NYCT Bus" and "NYCT Subway".
- **Heuristics:** Removed rows containing "this bus" (often generic), "bound track" (infrastructure, not service), and very short headers (less than 3 words).
- **Result:** A clean dataset of **227,210** unique alerts with %0 missing values (verified in MTA_eda_analysis.ipynb ).

### 2.2. Silver Label Generation (Weak Supervision)

To overcome the lack of labeled training data, I implemented a two stage and rule based pipeline to generate Silver labels.

#### Stage 1: Route Span Labeling

Code: 2_route_span_labeling.ipynb 

First, I identified the exact spans of **Routes** (e.g "Q44", "2 Train") mentioned in the text.

**My Method:** Agency specific Regex & Context Rules:

- **Bus:** Matches standard patterns (e.g Q, Bx, M prefixes) and handles suffixes (-SBS ).
- **Subway:** Distinguishes between Single Letters (A, Q), Digits (1-7), and Special codes (SIR, Shuttles). Crucially, filters false positives (e.g "E 149 St" is a street, not the E train).
- **Normalization:** Maps variants to canonical forms (e.g "Q44-SBS" to "Q44").
- **Stats:** I updated the column affected for approximately %4 of alerts where the text contained codes not present in the original alert.

#### Stage 2: Direction Labeling

Code: 3_direction_labeling_silver.ipynb 

Next, I detected and classified directions relative to the routes.

**My Design Choice:** I defined a strict **Priority Hierarchy** for labeling to resolve ambiguities:

1. **BOROUGH** ("Queens-bound") - Highest specificity.
2. **LOCAL** ("Uptown", "Downtown").
3. **COMPASS** ("Northbound", "Southbound").
4. **PLACE_BOUND** ("JFK-bound"):
- Used a **Leftward Scan Algorithm** starting from "bound".
- Includes **Abbreviation Awareness** (treats "St." as a token, not a stop).
- **Grammatical Rejection**: Filters out misleading phrases like "in both direction of".
1. **BOTH/EITHER** - Valid directionality for alerts affecting both sides.

**Outcome:** ~254k silver-labeled direction spans loaded into the RE pipeline.

### 2.3. Baseline Relation Extraction

Code: 4_baseline_re.ipynb 

The code to generate silver relation labels (linking Route to Direction).

- **My Design Choice:** A **Two Pass Segment Algorithm:**

**Pass 1 (Left-to-Right):** Tracks the Active Direction until a **Breakpoint** (newline, parenthesis, colon) resets the context.

**Pass 2 (Look Ahead):** Recovers unpaired routes by searching forward in the same segment (e.g., linking "M15" in *"M15 delays in both directions"*).

**EDA Statistics for RE:**

- **Coverage:** %79.4 of alerts contain extractable relations.
- **Total Pairs:** 293,474 relation pairs.
- **Distribution:** Southbound (%30.4) and Northbound (%29.8) dominate, but **Both Directions (%21.5)** is also a major category that required some attention.

### 2.4. Gold Dataset Creation

Code: 5_create_gold_dataset.py 

- **Status:** Ongoing annotation, haven't finished yet.
- **My Design Choice:** I stratified sampling by Complexity. As random sampling would be dominated by simple 1 Route, 1 Direction cases. 

I enforced a specific distribution:

- **Simple (%50):** less than or equal to 2 entities (e.g "M1 delays").
- **Moderate (%30):** 3-4 entities.
- **Complex (%20):** 5+ entities (e.g multiple routes/directions).
- Also ensured diversity of direction types within each case. Total samples: 600.

### 2.5. Neural Models

Codes: 6_bilstm_ner.ipynb, 6_deberta_ner.ipynb 

Trained on Silver data to learn the labeling logic.

- **My Design Choice for the Class Imbalance Handling:** The dataset is dominated by "O" tags (approximately %85). I implemented **Weighted Cross Entropy Loss** with specific boost factors:
    - I-ROUTE & I-DIRECTION: **1.5x boost** (to encourage continuity)
    - B-DIRECTION: **1.5x boost** (rare compared to routes)
    - B-ROUTE: **1.2x boost**

#### Model A: BiLSTM-CRF

Code: 6_bilstm_ner.ipynb 

I implemented a custom PyTorch model trained from scratch.

**1.Architecture Details:**

- **Embeddings:**
    - **Words:** 128-dim embeddings (Learned from scratch).
    - **Characters:** 50-dim embeddings -> **CharCNN** (50 filters, kernels [3,4,5]) to capture morphology (e.g. "-bound").
- **Encoder:** Bidirectional LSTM (Hidden Dim: **256**, Layers: **2**, Dropout: 0.3).
- **Decoder:** CRF (Conditional Random Field) for global sequence validity.

**2.Hyperparameters:**

- **Optimizer:** Adam (LR: 1e-3) with ReduceLROnPlateau.
- **Batch Size:** 64.
- **Loss:** Weighted Cross-Entropy (Fallback) / CRF Loss.
- **Epoch:** 3

#### Model B: DeBERTa-v3

Code: 6_deberta_ner.ipynb 

I fine-tuned microsoft/deberta-v3-base using a manual training loop.

**1.Architecture & Training:**

- **Model:** Pretrained Transformer with a Token Classification Head.

**2.Differential Learning Rates:**

- **Base Model:** 3e-5 (Preserve pretrained knowledge).
- **Classifier Head:** 1e-4 (Learn task fast).

**3.Optimization:**

- **Batch Size:** 128.
- **Scheduler:** Linear Warmup (%10 of steps).
- **Precision:** Mixed Precision (**FP16**) via GradScaler.
- **Gradient Clipping:** Norm 1.0 (Stability).
- **Epoch:** 3

## 3. Preliminary Results for NER on Silver Dataset

I compared the models on their ability to reproduce the Silver labels (Test set: %15 split).

Test scores:

| **Model**  | **Precision** | **Recall** | **F1-Score** |
| ---------- | ------------- | ---------- | ------------ |
| BiLSTM-CRF | 0.9973        | 0.9916     | **0.9944**   |
| DeBERTa-v3 | 0.9847        | 0.9993     | **0.9920**   |

My Analysis:

The F1 and other scores are basically perfect, but i think that is just because the data is very simple, and uses a very standard structured language. The regex labeling i used matches the text patterns exactly (i might have gone a bit on the feature engineering side). Since %72 of the alerts just have a direction right next to a route name, like "Northbound M15 delays," the model learns the pattern too easily. The rules for finding the direction handle over %80 of the cases just by looking at word proximity.

I'm suspecting the model might be memorizing the templates. It likely overfits to where the words sit in the sentence rather than understanding the text, especially in the %8 of alerts that are more confusing. 

When i test on the gold dataset with the harder examples, i think the scores will probably drop to something like 0.90. That should show if the model actually works when the simple regex rules do not apply.

## 4. Current Status

- Completed - **Data Pipeline:** Complete. Silver dataset is generated and robust.
- Completed - **Design:** Stratified sampling design is implemented to ensure rigorous evaluation.
- Completed - **Neural NER:** Implementation complete and validated on Silver data.
1. > Pending - **Gold Annotation:** Currently in progress (600 samples).
2. > Pending - Comparison of the gold dataset to heuristic RE.
3. > Pending - Using the gold dataset to compare the neural models and the weak supervision. 