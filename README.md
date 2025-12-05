# MTA_Project

# ANN Proposal

# **A Comparative Study of Neural Models for Direction Aware NER and Relation Extraction in Transit Service Alerts**

**Team: Onur Dursun** | Computer Engineering (MSc) | 25COMP5004 | Fall 2025

---

# Abstract

Public transit agencies publish real time service alerts as unstructured headlines that describe service changes, delays, and disruptions. A single alert often mixes multiple lines and directions in compact text, requiring downstream systems such as journey planners, chatbot assistants, and dashboards to extract structured facts about route identifiers, directional information, and route-direction relationships. This project compares neural models for direction-aware Named Entity Recognition (NER) and Relation Extraction (RE) on MTA transit alerts, evaluating a BiLSTM-CRF baseline against a DeBERTa-v3 transformer approach. Using a dataset of 230,012 alerts with 600 gold-annotated headers, I will assess model performance using the F1-score on the extracted (ROUTE, DIRECTION) pairs, with particular focus on challenging cases involving multiple routes and directions. Silver label weak supervision from regex and gazetteer rules will be used to pretrain models before gold fine-tuning, providing a scalable alternative to extensive manual labeling. Expected results should show DeBERTa-v3 an F1-score above 0.80 on the extracted pairs, representing a significant improvement over both BiLSTM-CRF (0.72-0.78) and rule-based approaches (0.60-0.65).

1. > **Problem Statement**

> Public transit agencies publish real time service alerts as unstructured headlines that describe service changes, delays, and disruptions. A single alert often mixes multiple lines and directions in compact text, for example:

1. > "Southbound Q65 and Q66 buses are running with delays"
2. > "95 St-bound R and Metropolitan Av-bound M trains are running express from Forest Hills 71 Av to Jackson Hts Roosevelt Av. Expect E F delays"

> Downstream systems such as journey planners, customer facing chatbot assistants, and real time dashboards need structured facts such as:

- > which specific route identifiers appear in the alert (ROUTE entities),
- > which directional information is present (DIRECTION entities),
- > and which direction applies to which route.

> **The Gap:** Most prior transit NER papers identify entities such as route codes and sometimes locations, but they do not model which direction is attached to which route, especially when alerts refer to multiple routes, implicit terminal based directions, or cases like “in both directions”. 

> Work on general event and relation extraction has introduced joint entity–relation architectures and end to end extraction frameworks, but these methods have not been applied to short, noisy transit alerts with dense route codes and “bound” expressions. There is also limited systematic comparison of classic sequence models and modern pretrained transformers on this type of domain specific, short text, direction aware task.

> **This Project:** I directly perform NER & RE by comparing baseline (BiLSTM-CRF) vs. advanced transformer models (DeBERTa-v3 base) on MTA transit alerts, using an  end-to-end F1-score on the extracted (ROUTE, DIRECTION) pairs and incorporating a comprehensive direction taxonomy that handles compass, borough, local, place bound, both directions, and unspecified directional information. 

> Relation extraction is evaluated and compared with three designs: a nearest direction heuristic baseline, a BiLSTM based pair classifier, and a DeBERTa v3 based classifier with entity markers, each predicting HAS_DIRECTION(ROUTE, DIRECTION) links over NER spans. Silver label weak supervision from regex and gazetteer rules over the full 230k alert corpus is used to pretrain both the NER and relation models before gold annotated fine tuning, providing a scalable alternative to extensive manual labelling and heavy LLM based augmentation.

> **Task type:** This project addresses supervised sequence labeling (NER) plus supervised binary relation classification on candidate entity pairs.

> **Input-output definition:** Input: a single alert header string. Outputs: (i) BIO tags for ROUTE and DIRECTION entities; (ii) a set of HAS_DIRECTION(ROUTE, DIRECTION) relations over the detected spans.

**A Note:** I will be working on the cross section of Intelligent Transportation Systems (ITS) and Artificial Intelligence for my thesis. A part of my thesis will be about NER on bus codes and location names, so this study will be foundational for me.

> ## **1.1 Literature Review**

> Before I start I must say that NER and RE work on specifically transit domain and bus codes is actually scarce, so to alleviate this I included out of domain research and I also wanted to look for researches with diverse methodologies to understand the application of NER and RE, which I have included below.

> Recent work shows that neural named entity recognition can reach strong performance in transport and engineering domains. Shou and Xu (2023) use a BERT MultiBiGRU CRF architecture to identify bus routes in noisy social media text and obtain high F1 for route codes, but they only predict route entities and do not attach directions or evaluate route–direction pairs. In a different research, Yang et al. (2024) construct an aviation product NER model that combines BERT, BiLSTM, and a CRF head and show that this compact stack performs well on long technical entities when guided by a domain ontology, then populate a Neo4j knowledge graph from the extracted spans. Both studies demonstrate that domain specialised NER with BERT plus sequence encoders is effective, but they stop at entity spans and do not address relation extraction or direction aware pairing.

> Beyond NER, the broader entity relation extraction literature explores how to connect entities into structured facts. Jiang et al. (2024) reformulate entity relation extraction as full shallow semantic dependency parsing: tokens become nodes in a graph, entity spans and relations are labeled edges, and the model performs joint inference with BERT plus BiLSTM encoders and a mean field variational CRF with second order edge interactions. This joint design reduces error propagation between NER and RE and yields state of the art results across news, scientific, and biomedical datasets, but it introduces cubic time inference and substantial implementation complexity.

> Zhang et al. (2025) take the opposite direction and design FastERE, a speed optimised pipeline framework for entity relation extraction. They first prune obvious negative entity pairs, then perform fast NER with adjacent attention and parallel relation classification, achieving near state of the art relation F1 on ACE04 and ACE05 *while* pruning more than 90 percent of candidate pairs and improving throughput by 6–20 times. This shows that carefully engineered pipelines can be competitive with heavier joint models when latency and throughput are critical.

> Phan et al. (2024) demonstrate a pragmatic, LLM assisted pipeline for biomedical entity relation extraction on the BioRED benchmark. They keep classic NER components, but use GPT 4 to augment training data and Gemini to generate relation oriented explanations, which are then fed into a compact classifier. This combination of data augmentation and LLM guided classification substantially improves F1 over strong baselines when labels are few and relations are ontology heavy.

> Across these lines of work, three design themes are most relevant for this project:

- > NER with BERT plus sequence encoders and CRF decoding for transit domain specific texts (Shou & Xu, 2023; Yang et al., 2024)
- > Joint versus pipeline strategies for entity relation extraction, including graph based joint parsing (Jiang et al., 2024) and fast pruned pipelines (Zhang et al., 2025)
- > Augmentation and weak supervision schemes that combine classic architectures with external signal such as LLMs or large silver label corpora (Phan et al., 2024).



---



---

---

---

2. Research Questions

**RQ1 - (NER)**

- Do transformer based models built on DeBERTa v3 significantly outperform a BiLSTM CRF baseline for direction aware NER on short, entity dense transit alerts, both overall and for challenging direction categories such as PLACE_BOUND and BOTH_DIRECTIONS?

**RQ2 - (Relation extraction)**

- Given NER predictions, does a DeBERTa v3 based relation classifier achieve higher Set F1 on (ROUTE, DIRECTION) pairs than a BiLSTM based relation classifier and a simple nearest direction heuristic, especially in multi route and multi direction headers?

**RQ3 - (Training strategy)**

- Does silver label pre training on a large corpus of automatically labelled alerts improve NER and RE performance for both architectures, and does it change the relative gap between BiLSTM CRF and DeBERTa v3 based models?

---

3. Dataset

**Source:** MTA (Metropolitan Transportation Authority of NYC) Service Alerts from [NYC Open Data](https://data.ny.gov/Transportation/MTA-Service-Alerts-Beginning-April-2020/7kct-peq7/about_data)

**Volume:** Around 400,000 raw records including Subway and Trains; 600 gold-annotated headers

**Structure:** Unstructured alert header text with semi-structured route lists. Bus, Subway codes entirely labeled.

**License / use:** NYC Open Data terms, it is suitable for research use.

**Granularity:** One row per unique alert header after preprocessing and deduplication.

| **Characteristic** | **Details**                                     |
| ------------------ | ----------------------------------------------- |
| Text length        | 10-30 tokens (short, entity-dense)              |
| Entity types       | ROUTE, DIRECTION                                |
| Complexity         | Single/multi-route; explicit/implicit direction |
| Splits             | Temporal (70/15/15) + Route-stratified (80/20)  |

## **3.1 Scope**

**Final preprocessed corpus**

- **230,012 alerts**
- **Columns used:** Alert ID, Date, Agency, Status Label, Affected, Header
- **Transit agency split:** Approx. 72.5 percent Subway, 27.5 percent Bus

**Gold subset**

- **600 headers** manually annotated with:
    - ROUTE and DIRECTION entities (with subtypes: COMPASS, BOROUGH, LOCAL, PLACE_BOUND, BOTH_DIRECTIONS, UNSPECIFIED)
    - HAS_DIRECTION(ROUTE, DIRECTION) relations

---

## **3.2 Preprocessing**

Pipeline implemented in my current Data Preprocessing code:

- Filter alerts to NYCT Bus and NYCT Subway lines.
- Drop unused columns (e.g description fields) and rows with missing Header.
- Remove:
    - duplicate headers,
    - very short headers (too few words/characters),
    - patterns like "this bus" and "bound track" that are used specifically for onboard screen alerts.
- Normalise Affected column (pipe separated list of routes) into a machine friendly list format.
- Export as MTA_Data_preprocessed.csv for all EDA and training.

> Normalization is minimal (Unicode and whitespace cleanup without lowercasing) to preserve route codes. BiLSTM models use whitespace + punctuation tokenization with FastText + char embeddings; DeBERTa-v3 uses its native subword tokenizer. No aggressive text augmentation will be used on gold data; instead, i will rely on silver-label weak supervision over the full 230k alerts as the primary form of augmentation, with optional light token-level noise only during silver pretraining.

---

## **3.3 Direction statistics summary (EDA)**

From rule based extraction on the preprocessed corpus:

- **Total direction labels:** Around 254k
- Dominant types:
    - **SOUTHBOUND + NORTHBOUND:** 46 percent combined
    - **UNSPECIFIED:** 19 percent
    - **BOTH_DIRECTIONS:** 17 percent
    - **PLACE_BOUND:** 10 percent
    - Remaining 8 percent spread across EAST/WEST and BOROUGH/LOCAL labels (e.g. MANHATTAN_BOUND, UPTOWN, etc.)

![direction_categories_grouped.png](ANN%20Proposal.assets/direction_categories_grouped.png)



**Directions per alert**

- 0 directions: 21 percent of alerts
- 1 direction: 71 percent
- 2 or more directions: 8 percent

These distributions drive the evaluation breakdowns (single vs multi-direction, rare direction types).

---

## **3.4 Route and alert complexity (summary)**

- Most alerts mention **one route**, with a long tail of multi-route headers (e.g. “Q65 and Q66 buses…”).
- Hard instances are:
    - multi-route + multi-direction headers,
    - implicit PLACE_BOUND directions,
    - alerts with routes but no explicit direction (UNSPECIFIED).

![exact_route_distribution.png](ANN%20Proposal.assets/exact_route_distribution.png)

---

## **3.5 Splits and data quality**

**Splits**

- Base split at alert level:
    - **Train:** 70 percent
    - **Dev:** 15 percent
    - **Test:** 15 percent
- Same split used for all models; additional route-aware split planned for generalisation analysis.

**Quality / imbalance notes**

- Headers with missing text are removed; remaining core fields are complete.
- Silver labels (from regex + gazetteers) are noisy especially for PLACE_BOUND, (e.g Metropolitan Av-Bound is marked as "Av-Bound") however this only affects the name of the Place_bound directions instead of the detection itself; and they are used only for pretraining, with final evaluation on gold labels.
- Class imbalance:
    - Direction types are skewed toward COMPASS, UNSPECIFIED, BOTH_DIRECTIONS.
    - RE might be imbalanced (many NO_RELATION pairs will exist).
- Mitigation:
    - class weighting or negative downsampling in RE,
    - per-category reporting for DIRECTION subtypes and for simple vs complex alerts.



---

3. Direction Types

Transit alerts in the dataset contain several types of directional information:

| **Direction Type**         | **Example Alert**                                                                                                                 | **Pseudo Structured Output**                                 |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Explicit (Compass)         | "Southbound Q65 buses are delayed"                                                                                                | (Q65, SOUTHBOUND)                                            |
| Explicit (Borough)         | "Manhattan-bound M60 buses"                                                                                                       | (M60, MANHATTAN_BOUND)                                       |
| Explicit (Local)           | " Downtown 2 trains are delayed while our crews investigate a switch malfunction at 96 St."                                       | (2, DOWNTOWN)                                                |
| Explicit (Place-Bound)     | "Jamaica-bound F trains are running at slower speeds while we address a signal problem near Broadway-Lafayette St."               | (F, PLACE_BOUND)                                             |
| Explicit (Both-Directions) | "E trains are running with delays in both directions."                                                                            | (E, BOTH_DIRECTIONS)                                         |
| Implicit**(Excluded)*      | "The 4:00 PM BxM6 bus trip scheduled to depart Metropolitan Ave/Unionport Rd will not run."                                       | (BxM6, IMPLICIT)                                             |
| Ambiguous/Multi            | "Southbound S78, S79-SBS, and westbound SIM1C buses will not stop at Hylan Blvd/Bay Terrace because of road work at the location" | "Southbound" = [S78, S79-SBS]<br/><br/>"Westbound" = [SIM1C] |
| Unspecified                | "You may wait longer for a G train. We're running as much service as we can with the train crews we have available."              | (G, UNSPECIFIED)                                             |

*Note: Implicit directions will not be annotated as DIRECTION entities in the current gold or silver data; as they can not be easily auto-labeled since each departure location has a different street/geographical name.

4. **Direction Taxonomy**

*The project will use a comprehensive taxonomy to normalize directional expressions:*

1. COMPASS Directions: (NORTHBOUND, SOUTHBOUND, EASTBOUND, WESTBOUND)

2. BOROUGH Directions: (MANHATTAN_BOUND, QUEENS_BOUND, BRONX_BOUND, BROOKLYN_BOUND, STATENISLAND_BOUND)

3. LOCAL Directions: (UPTOWN, DOWNTOWN)

4. PLACE-BOUND Directions: (LaGuardia-Bound etc.)

5. UNSPECIFIED: (No explicit directional information present in the alert)

---

5. **Methodology**

## **5.1 Method (ANN Design)**

- **Architecture:** I will compare two distinct pipelines for Named Entity Recognition (NER) and Relation Extraction (RE).
    - **Baseline (BiLSTM-based):** For NER, I'll feed FastText word embeddings and character level embeddings into a two layer BiLSTM with a CRF decoding layer. For RE, I will use the same encoder but classify pairs using a Multilayer Perceptron (MLP) that looks at hidden states, direction type, and distance embeddings.
    - **Advanced (DeBERTa-based):** I will build the NER component on DeBERTa v3 base followed by a simple linear classification head (Softmax). For RE, I'll use a separate DeBERTa v3 instance that inserts entity markers (e.g. [ROUTE], [/ROUTE]) around spans and uses the [CLS] token for classification.
- **Loss:** I will optimize the BiLSTM-CRF baseline using negative log-likelihood, whereas the DeBERTa model will be optimized using standard token-level Cross-Entropy Loss. For RE, I'll use binary cross entropy loss on candidate pairs (HAS_DIRECTION vs NO_RELATION), applying class weights to handle imbalance.
- **Optimizer & schedule:** I will use the Adam optimizer (LR 1e-3) for the BiLSTM models with a reduce on plateau schedule. For DeBERTa, I'll use AdamW (LR 3e-5 for encoder, 1e-4 for heads) with a linear warmup for the first 10 percent of steps.
- **Regularization:** I plan to use dropout (0.3 for BiLSTM, 0.1 for DeBERTa), weight decay, gradient clipping, and early stopping based on validation Set F1.
- **Initial hyperparameters:** I'll set batch sizes to 32 for BiLSTM and 16 for DeBERTa. Training will run for up to 30 epochs for silver pre training and 10 to 20 epochs for gold fine tuning.

## **5.2 Baselines**

- **Comparisons:** I will compare my DeBERTa v3 pipeline against a BiLSTM CRF baseline and a non neural rule based baseline (Regex or Gazetteer NER plus Nearest Direction heuristic).
- **Success criterion:** My goal is for the DeBERTa v3 pipeline to beat the BiLSTM baseline by at least 3 to 5 percentage points in Set F1 and significantly outperform the rule based approach.

## **5.3 Experimental Plan**

- **Protocol:** I will use a temporal split (70 percent train, 15 percent dev, 15 percent test) on the 230012 alerts. I'll also run a secondary route stratified split (80 or 20) to test generalization on rare routes.
- **Ablation & Training Strategy:** To isolate the sources of performance gain, I will conduct a data ablation study comparing 'Gold Only' training against 'Silver Pre-training + Gold Fine-tuning'. Additionally, for the Relation Extraction component, I will ablate the entity markers to compare the performance of [CLS] only classification versus input augmented with explicit [ROUTE] and [DIRECTION] boundary tokens.
- **Compute budget:** I will use my local RTX 3060 Ti and Google Colab A100s. I estimate the budget will be under 8 GPU hours.

## **5.4 Evaluation**

- **Primary metrics:** My main metric for comparison will be the F1-score on the extracted (ROUTE, DIRECTION) pairs, treating predicted and gold relations as sets of tuples.
- **Component metrics:** I will also track:
    - **NER:** Span-level precision, recall, and F1 for ROUTE and DIRECTION. I'll specifically look at per-category F1 for direction subtypes (e.g. COMPASS, BOROUGH, PLACE_BOUND, BOTH_DIRECTIONS).
    - **RE:** Precision, recall, and F1 for HAS_DIRECTION detection, evaluated both on gold entities (upper bound) and end-to-end with predicted entities.
- **Complexity-aware breakdowns:** To understand performance on harder tasks, I will break down results by:
    - Single-route vs. multi-route headers.
    - Alerts with zero, one, or multiple directions.
    - Frequent vs. rare routes (using the route-stratified split).
- **Diagnostics:** I plan to use confusion matrices to see which direction subtypes are confused and analyze HAS_DIRECTION vs. NO_RELATION errors. I'll also check precision-recall curves for threshold sensitivity and perform a qualitative error analysis on implicit PLACE_BOUND cases and "both directions" language.

## **5.5 Anticipated Results**

- **Metric name:** F1 score on (ROUTE, DIRECTION) pairs (End-to-End Pair Performance) and Span-F1.
- **Where measured:** Held-out test set (gold annotations).
- **Exact goals:**
    - **Rule-based:** Around 0.60–0.65 F1-score on pairs.
    - **BiLSTM-CRF:** 0.72–0.78.
    - **DeBERTa-v3:** More than **0.80**, aiming for **0.83–0.85** after silver pre-training.
- **NER Specifics:** I expect ROUTE F1 to be more than 0.95 for both models. For DIRECTION F1, I anticipate BiLSTM reaching 0.88–0.90 and DeBERTa exceeding 0.92, with the biggest gains in the PLACE_BOUND and BOTH_DIRECTIONS categories.
- **Hypothesis:** I expect DeBERTa-v3 to retain clear advantages in multi-route/multi-direction headers and implicit directions. While silver pre-training should narrow the gap between architectures, I don't expect it to eliminate it.

# **References**

1. Shou, Y., & Xu, J. (2023). A Novel Named Entity Recognition Mehod for Bus Route Identification in Social Media. *2023 2nd International Conference on Machine Learning, Cloud Computing and Intelligent Mining (MLCCIM)*, 59–64. [https://doi.org/10.1109/mlccim60412.2023.00014](https://doi.org/10.1109/mlccim60412.2023.00014)
2. Lin, J., & Liu, E. (2022). Research on Named Entity Recognition Method of Metro On-Board Equipment Based on Multiheaded Self-Attention Mechanism and CNN-BiLSTM-CRF. *Computational Intelligence and Neuroscience*, *2022*, 1–13. [https://doi.org/10.1155/2022/6374988](https://doi.org/10.1155/2022/6374988)
3. Mahmood, T., Mujtaba, G., Shuib, L., Zulfiqar Ali, N., Bawa, A., & Karim, S. (2017). Public bus commuter assistance through the named entity recognition of twitter feeds and intelligent route finding. *IET Intelligent Transport Systems*, *11*(8), 521–529. [https://doi.org/10.1049/iet-its.2016.0224](https://doi.org/10.1049/iet-its.2016.0224)
4. Zulkarnain, & Tsarina Dwi Putri. (2021). Intelligent transportation systems (ITS): A systematic review using a Natural Language Processing (NLP) approach. *Heliyon*, *7*(12), e08615–e08615. [https://doi.org/10.1016/j.heliyon.2021.e08615](https://doi.org/10.1016/j.heliyon.2021.e08615)
5. Yang, M., Namoano, B., Farsi, M., & Ahmet Erkoyuncu, J. (2024). Named Entity Recognition in Aviation Products Domain Based on BERT. *IEEE Access*, *12*, 189710–189721. [https://doi.org/10.1109/access.2024.3516390](https://doi.org/10.1109/access.2024.3516390)
6. Jiang, S., Li, Z., Zhao, H., & Ding, W. (2024). Entity-Relation Extraction as Full Shallow Semantic Dependency Parsing. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, *32*, 1088–1099. [https://doi.org/10.1109/taslp.2024.3350905](https://doi.org/10.1109/taslp.2024.3350905)
7. Phan, C.-P., Phan, B., & Chiang, J.-H. (2024). Optimized biomedical entity relation extraction method with data augmentation and classification using GPT-4 and Gemini. *Database*, *2024*. [https://doi.org/10.1093/database/baae104](https://doi.org/10.1093/database/baae104)
8. Zhang, W., Xu, T., Hua, Y., Feng, Z., & Song, X. (2025). Fastere: a fast framework for entity relation extractions. *Data Mining and Knowledge Discovery*, *39*(6). [https://doi.org/10.1007/s10618-025-01146-y](https://doi.org/10.1007/s10618-025-01146-y)