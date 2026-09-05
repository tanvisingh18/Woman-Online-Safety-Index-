# Chapter 2. Literature Review

This chapter surveys six bodies of work that ground the dual-index design used in this project: the **Women Harassment Severity Index (WHSI)** for user-side harm, the **Women-Targeted Speech Harm Index (WTSHI)** speaker-role construct that feeds it, and the **Moderation Responsiveness Index (MRI)** for platform accountability. Section 2.1 reviews women-targeted harassment and the models built to detect it. Section 2.2 traces general hate-speech and toxicity detection architectures. Section 2.3 covers datasets and benchmarks. Section 2.4 examines platform-level analysis. Section 2.5 reviews moderation and accountability. Section 2.6 reviews fuzzy inference, continuous scoring, and explainability — the methodological basis of WHSI. Section 2.7 synthesises the research gap.

---

## 2.1 Women-Targeted Harassment and Online Abuse

Online misogyny is not a niche subtype of generic toxicity: it is a patterned form of harm with its own taxonomies, datasets, and failure modes. Guest et al. (2021) showed that untrained crowdsourced annotators struggle to identify misogyny on Reddit and that a hierarchical, expert-labelled scheme is required; Kirk et al. (2023) then made fine-grained sexism (threats, derogation, animosity, and related subtypes) a shared-task standard through EDOS. Those findings matter for this project because a platform-level women’s safety score that treats all “toxic” language as equivalent will mis-count both perpetrator attack and victim disclosure. The remainder of this section reviews the women-specific detection systems that have been proposed to operationalise that problem — transformers, hybrids, multimodal meme models, and modular specialists — and the datasets and metrics they use.

### Existing models and techniques

Recent research has prioritized the deployment of transformer-based architectures for detecting sexist and derogatory content, capitalizing on their robust contextual processing capabilities. Prithila et al. [1] conducted a comparative analysis of five transformer models—BanglaBERT, XLM-RoBERTa, m-BERT, DistilBERT, and Bangla-Bert-Base—for the binary classification of misogynistic comments in Bengali and English. The models were fine-tuned using a standard architecture that incorporated embedding, encoder, linear, and softmax layers. The results indicated that Bangla-BERT-base achieved peak performance on the Bengali dataset with an F1-score of 94%, whereas m-BERT demonstrated superior efficacy on the English dataset with an F1-score of 86.1% [1].

Martinez et al. [2] developed a hybrid system for detecting online sexism in Gab and Reddit posts by implementing a feature-union approach. This method integrates 27 lexical features with transformer-based representations extracted from the Twitter-RoBERTa-base-sentiment-latest model. By employing a voting classifier ensemble comprising nine machine learning models, the authors observed that this feature union strategy enhanced binary classification accuracy by approximately 20% compared with models relying exclusively on lexical features [2].

Abburi et al. [3] introduced a semi-supervised, multi-level neural architecture designed for fine-grained, multi-label sexism classification. Their framework integrates domain-adapted BERT with biLSTM and attention mechanisms and synthesizes word embeddings from ELMo, GloVe, and tBERT. To address the challenges associated with textual diversity and class imbalance, the authors implemented self-training-based data augmentation. Furthermore, they devised a confidence-modified binary cross-entropy loss function and a sequential training strategy that classifies sexism into 8, 15, and 23 categories [3].

Mohasseb and Amer [4] proposed a framework to improve misogyny detection through sentence-level semantic enrichment utilizing MiniLM, FastText, and ConceptNet. Their methodology generates semantically meaningful paraphrases of the input text, which are then evaluated across traditional, deep learning, and transformer-based models. The study concluded that the integration of ELECTRA with ConceptNet yielded the highest performance, achieving an F1-score of 92% and an accuracy of 97% [4].

Akinduyite and Chris-Umoru [5] developed a multimodal deep learning framework, the BiLSTM-VGG16 architecture, for the automatic detection of misogyny in Internet memes. Their system executes independent classification processes: Bidirectional Long Short-Term Memory for textual data and Visual Geometry Group 16 for image data, without utilizing multimodal fusion. Evaluated on the MAMI dataset, this approach achieved an accuracy of 0.99 for both text and image modalities [5].

Recent scholarship has increasingly transitioned from unimodal systems to complex hybrid and multimodal architectures. Rehman et al. [8] introduced the ASCEND framework, which employs an adaptive threshold-based supervised contrastive learning mechanism utilizing RoBERTa embeddings. To improve the identification of implicit sexism, this model incorporates word-level attention and integrates auxiliary features, including sentiment, emotion, and toxicity [8]. Similarly, Yadavalli and Sahoo [10] proposed a multigranular hybrid architecture that combines word-level BERT embeddings with character-level features from RoBERTa. Their model employs CNNs for local pattern extraction, Bi-LSTM networks for capturing sequential dependencies, and self-attention mechanisms to model global contextual relationships [10].

Multimodal meme detection has also advanced significantly. Singh et al. [6] developed a BERT*-based model that combines text pre-trained on hateful meme content with a Vision Transformer via an attention-based approach [6]. Karishma and Akila [7] investigated the efficacy of semi-supervised learning algorithms by specifically comparing self-training, co-training, and mean student-teacher models within a Gated Recurrent Unit architecture utilizing GloVe embeddings [7]. To mitigate data scarcity, Rodriguez-Sanchez et al. [9] utilized a multilingual XLM-R transformer enhanced with a Sentence-BERT layer, leveraging unsupervised task adaptation and semisupervised pseudo-labeling [9].

Kaware and Raut [11] presented a machine learning framework for the automatic detection of misogynistic content in multilingual social media data, specifically targeting Hindi and Indian English. Their approach utilized hybrid linguistic features, combining bag of words, TF-IDF, and VADER-based sentiment polarity scores, which were processed by four classifiers: Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machine. Through 10-fold cross-validation with grid-search hyperparameter tuning, the Random Forest classifier achieved optimal results, attaining 96% accuracy on the English dataset and 93% on the Hindi dataset [11].

Altarawneh et al. [12] introduced AD-ASH, a modular framework designed to address the limitations inherent in monolithic AI architectures when classifying complex, multi-label online harassment. Observing that no single architecture excels across all sub-types, the authors found that fine-tuned BERT performed best for "commenting," the CNN-RNN architecture was most effective for "ogling," and fine-tuned DeepSeek proved optimal for "groping." The AD-ASH system dynamically assigns each harassment category to the best-performing specialist model, achieving an exact match ratio of 66.0% and a Hamming Score of 85.3%, thereby surpassing monolithic baselines [12].

### Datasets used in prior vs current work

The datasets used in these studies vary significantly in terms of volume, language, and annotation granularity. Prithila et al. [1] curated a Bengali dataset by synthesizing four existing hate speech corpora: BD-SHS, the Bengali Online Comments Dataset, ToxLenbn, and bnhatespeech. For English, they integrated an aggregated labeled dataset of sexist and non-sexist comments with the ISEP Sexism dataset [1].

Martinez et al. [2] leveraged the SemEval 2023 Task 10 dataset for binary sexism detection. Abburi et al. [3] utilized a substantial labeled dataset for multi-label sexism classification, annotated with multiple categories including role stereotyping, body shaming, and threats.

Mohasseb and Amer [4] employed a dataset of Reddit entries annotated to differentiate between misogynistic and non-misogynistic content. Each entry is accompanied by metadata with hierarchical annotations supporting both binary classification and fine-grained subtype identification [4]. The Multimedia Automatic Misogyny Identification dataset was used by Akinduyite and Chris-Umoru [5], Singh et al. [6], and Karishma and Akila [7]. This dataset supports two subtasks: binary classification and four-way fine-grained categorization (stereotypes, objectification, shaming and violence). While some studies leveraged the full dataset, others utilized subsets for training [5].

Rehman et al. [8] conducted their fine-grained classification using the EXIST 2021 dataset and MLSC, while Rodriguez-Sanchez et al. [9] incorporated both the EXIST 2021 and 2022 datasets, which provide tweets annotated for sexism identification and categorization [9]. Yadavalli and Sahoo [10] utilized the Hate Speech and Offensive Language Dataset, which they re-annotated into a binary classification format [10]. Kaware and Raut [11] worked with the TRAC-2 benchmark dataset, which consists of text instances in Hindi and Indian English across training, validation, and test partitions [11]. Finally, Altarawneh et al. [12] assessed their frameworks using the SafeCity dataset, which includes anonymized sexual harassment narratives [12].

For this project, the operative women-specific training and holdout anchor is Kirk et al.’s EDOS corpus (SemEval-2023 Task 10), discussed in Section 2.3 — not MAMI, EXIST, or SafeCity. The studies above establish that women-targeted detection is a distinct modelling problem with its own benchmarks; they do not yet provide a platform-level aggregation method.

### Evaluation metrics

The evaluation metrics selected in these studies corresponded to the requirements of their respective classification tasks. Prithila et al. [1] assessed their binary classification models using accuracy, precision, recall, and macro F1-score. Martinez et al. [2] similarly utilized F1-score, precision, and recall to evaluate their binary sexism detection system. For multi-label classification, Abburi et al. [3] adopted micro-F1, macro-F1, weighted-F1, precision and recall. Mohasseb and Amer [4] employed F1-score, recall, and accuracy to evaluate the impact of their enrichment methods. Akinduyite and Chris-Umoru [5] reported the accuracy and F1-score for their multimodal system. Rehman et al. [8] and Rodriguez-Sanchez et al. [9] prioritized Macro-F1 for fine-grained categorization. Singh et al. [6] and Karishma and Akila [7] utilized precision, recall, and F1-score for multimodal meme classification (MC). Kaware and Raut [11] evaluated their framework using accuracy derived from 10-fold cross-validation with grid-search hyperparameter tuning. Altarawneh et al. [12] employed the Exact-Match Ratio and Hamming Score to assess their multi-label classification framework.

### Key findings

The reviewed literature highlights significant advancements in model performance and the efficacy of various methods. Prithila et al. [1] reported that the Bangla-Bert-Base model surpassed all comparative benchmarks on the Bengali dataset, achieving an F1-score of 94%, an 8% improvement over preceding results. While DistilBERT demonstrated the least efficacy on Bengali data, attributed to its monolingual English pre-training, multilingual alternatives such as XLM-RoBERTa and m-BERT yielded competitive results [1].

Martinez et al. [2] observed a 20% augmentation in classification accuracy relative to their baseline lexical-only methodology, with a nine-classifier ensemble voting system attaining a peak test F1-score of 0.77, precision of 0.77, and recall of 0.79 [2]. Abburi et al. [3] demonstrated that their semi-supervised architecture surpassed multiple baselines and state-of-the-art models across several metrics. By integrating a multilevel training framework with the CBCE loss function, the authors effectively mitigated the challenges posed by the scarcity of labeled data [3].

Mohasseb and Amer [4] determined that semantic enrichment substantially enhances classification performance across diverse model architectures, with ConceptNet providing the most pronounced gains; specifically, the integration of ELECTRA and ConceptNet yielded the highest performance [4]. Akinduyite and Chris-Umoru [5] reported notable accuracy improvements using a BiLSTM-VGG16 architecture on the MAMI dataset, which was effective across both textual and visual modalities [5].

Rehman et al. [8] indicated that implementing adaptive thresholding within contrastive learning outperformed established baselines and large language models by a margin of up to 36.14% in macro-F1 [8]. Yadavalli and Sahoo [10] demonstrated that the fusion of word- and character-level features results in superior performance, attaining an accuracy of 98.38% and an F1-score of 97.81% [10]. Singh et al. [6] indicated that pre-training BERT on datasets containing hateful memes facilitates a performance increase of more than 10% in misogyny identification tasks [6]. Karishma and Akila [7] established self-training as the most efficacious semi-supervised learning algorithm, reporting a testing accuracy of 79.64% and an F1-score of 79.59% [7]. Rodriguez-Sanchez et al. [9] established a new state-of-the-art performance for the EXIST 2021 and 2022 datasets, achieving 81.2% accuracy in sexism identification via task adaptation [9].

Kaware and Raut [11] evidenced that hybrid feature engineering—integrating lexical, statistical, and sentiment-based representations—outperforms methodologies utilizing single feature types; their Random Forest model attained 96% accuracy on the English dataset and 93% on the Hindi dataset. Finally, Altarawneh et al. [12] concluded that no monolithic architecture consistently excels across all harassment subtypes, demonstrating that their modular AD-ASH system outperformed the leading monolithic baselines.

### Research gap and limitations

Each study identified unique constraints and offered potential avenues for future research.

Prithila et al. [1] highlighted that language bias in multilingual models can stem from data scarcity, translation inaccuracies, and code-switching challenges. Additionally, they noted potential inconsistencies in the Bengali dataset, which originated from disparate sources with different annotation standards. Martinez et al. [2] pointed to computational constraints and stressed the necessity of implementing continuous learning algorithms to effectively process real-time data streams. Abburi et al. [3] observed that even with semi-supervised augmentation, labeled data remains scarce for specific classes within their 23-category framework; they also highlighted the heavy computational demands of their approach.

Mohasseb and Amer [4] cited dataset imbalance as a key issue, suggesting that future efforts utilize cost-sensitive learning or synthetic oversampling to mitigate this. Akinduyite and Chris-Umoru [5] remarked that their system’s reliance on text and image data excludes audio or video, underscoring the importance of developing advanced fusion techniques for a more cohesive multimodal integration.

Rehman et al. [8] observed that because sexism is deeply tied to cultural contexts and personal subjectivity, model generalizability remains restricted across diverse social environments. Yadavalli and Sahoo [10] and Rodriguez-Sanchez et al. [9] indicated that many existing models concentrate on English or specific social platforms, failing to adequately support diverse linguistic environments or code-mixed text [10], [9]. Singh et al. [6] and Rodriguez-Sanchez et al. [9] identified that detecting nuanced elements such as sarcasm, irony, and intricate visuo-linguistic cues in memes presents a notable difficulty, frequently resulting in false negatives. Furthermore, Yadavalli and Sahoo [10] and Karishma and Akila [7] highlighted persistent concerns regarding the quality of unlabeled data in semi-supervised learning and the substantial computational expense required to deploy large-scale transformer models.

Kaware and Raut [11] recognized that their reliance on a single benchmark dataset and the exclusion of deep learning techniques constituted significant limitations. Altarawneh et al. [12] performed a label-quality assessment that uncovered major inconsistencies in their dataset, including high mismatch rates for "commenting" (22.0%) and "ogling" (21.5%), alongside a notable directional bias for "groping." These findings emphasize the necessity of strict annotation protocols for developing sensitive AI systems.

### Comparative analysis

| Ref | Authors | Model/Technique | Dataset | Evaluation Metrics | Key Results | Research Gaps/Limitations |
|-----|---------|-----------------|---------|--------------------|-------------|---------------------------|
| [1] | Prithila et al. | BanglaBERT, XLM-RoBERTa, m-BERT, DistilBERT, Bangla-Bert-Base | Bengali (30,853 texts), English (15,138 statements) | Accuracy, Precision, Recall, F1-score | Bangla-Bert-Base: 94% F1 (Bengali); m-BERT: 86.1% F1 (English) | Language bias in multilingual models; inconsistent annotation across source datasets |
| [2] | Martinez et al. | Feature union (lexical + transformer), voting classifier ensemble | SemEval 2023 Task 10 (20,000 samples) | F1-score, Precision, Recall | 20% accuracy improvement; best F1-score 0.77 | Computational constraints; need for real-time adaptation |
| [3] | Abburi et al. | Semi-supervised multi-level neural (tBERT + biLSTM + attention) | Everyday Sexism Project (13,023 labeled, 70,000 unlabeled) | Micro-F1, Macro-F1, Weighted-F1, Precision, Recall | Outperforms baselines; effective multi-label classification | Scarcity of labeled data for rare categories; high computational demand |
| [4] | Mohasseb and Amer | Semantic enrichment (MiniLM, FastText, ConceptNet) + multiple classifiers | Reddit dataset (6,567 entries) | F1-score, Recall, Accuracy | ELECTRA+ConceptNet: 92% F1, 97% accuracy; BiLSTM: 87% F1 | Dataset imbalance; need for larger, more diverse datasets |
| [5] | Akinduyite and Chris-Umoru | BiLSTM (text) + VGG16 (image) multimodal | MAMI dataset (12,000 memes) | Accuracy, F1-score | 0.99 accuracy on text and image data | Limited to text and image; no modal fusion; audio/video not considered |
| [6] | Singh et al. | BERT*+ViT (Hate-pretrained) | MAMI | Prec, Rec, F1 | >10% improvement on Subtask A | Complex visuo-linguistic cues; high complexity |
| [7] | Karishma and Akila | GRU + SSL (Self-training) | MAMI | Acc, Prec, Rec, F1 | 79.64% testing Acc; 79.59% F1 | Unlabeled data quality; robustness to noise |
| [8] | Rehman et al. | ASCEND (Contrastive Learning + Attention) | EXIST 2021, MLSC | Macro-F1 | 36.14% improvement over baselines | Cultural subjectivity; annotation noise |
| [9] | Rodriguez-Sanchez et al. | XLM-R + SBERT (Task Adaptation) | EXIST 2021, 2022 | Acc, Prec, Rec, F1 | 81.2% Acc (EXIST 2021); 61.5% F1 (Task 2) | Sarcasm/irony detection; pseudo-labeling bias |
| [10] | Yadavalli and Sahoo | Hybrid BERT-RoBERTa (CNN + Bi-LSTM + SA) | Abusive Language Dataset | Acc, Prec, Rec, F1 | 98.38% Acc; 97.81% F1 | Monolingual focus; limited socio-linguistic cues |
| [11] | Kaware and Raut | ML classifiers (RF, SVM, LR, NB) with hybrid features (BoW, TF-IDF, VADER) | TRAC-2 (6,529 instances) | Accuracy (10-fold CV) | 96% Acc (English); 93% Acc (Hindi) | Single dataset reliance; no deep learning methods |
| [12] | Altarawneh et al. | AD-ASH (modular specialist models: BERT, CNN-RNN, DeepSeek) | SafeCity (9,892 narratives) | Exact-Match Ratio, Hamming Score | 66.0% EMR; 85.3% Hamming Score | Label inconsistencies in dataset; directional annotation bias |

### Overall research gaps (Section 2.1)

A synthesis of the reviewed literature reveals several recurring research gaps that require further investigation.

1. **Scarcity of language resources.** The literature consistently underscores the scarcity of labeled datasets for low-resource languages, notably Bengali and Hindi. Although Prithila et al. [1] attempted to mitigate this issue via dataset aggregation, the resulting inconsistencies in annotation protocols remain a significant hurdle.

2. **Computational and scalability constraints.** Both Martinez et al. [2] and Mohasseb and Amer [4] identified the substantial computational demands of transformer-based architectures.

3. **Adaptability to dynamic content.** Martinez et al. [2] emphasized the evolving nature of online sexist discourse. The current lack of continuous learning mechanisms poses a significant barrier to maintaining the efficacy of models against emerging linguistic patterns.

4. **Multimodal and cross-modal integration.** Despite the demonstrated utility of multimodal approaches, Akinduyite and Chris-Umoru [5] and Singh et al. [6] identified a lack of robust fusion techniques in multimodal data integration.

5. **Semantic enrichment and contextualization.** Mohasseb and Amer [4] illustrated that semantic enrichment enhances the detection of implicit misogyny; however, the generalizability of these techniques across diverse cultural and linguistic contexts remains under-researched.

6. **Multi-label and fine-grained classification.** Abburi et al. [3] pioneered multi-label classification frameworks (23 categories); however, the scarcity of labeled data for infrequent classes persists.

7. **Cross-domain and cross-linguistic generalizability.** Prithila et al. [1] highlighted performance disparities in monolingual models across varying linguistic contexts.

8. **Detection of nuanced linguistic cues.** Models frequently struggle to identify sarcasm, irony, and implicit cues, particularly in multimodal memes [6], [9].

9. **Human subjectivity and annotation bias.** Dataset inconsistencies are caused by the inherent subjectivity surrounding sexism, which adversely affects model reliability. Altarawneh et al. [12] highlighted label inconsistencies in existing benchmarks [8], [9], [12].

10. **Operational efficiency.** Balancing high predictive accuracy with the computational constraints required for real-time moderation continues to be a persistent challenge [6], [10].

None of these studies aggregate comment-level sexism detections into a platform-level women’s harm score, and none separate perpetrator attack from victim disclosure before scoring. Those two absences are the design rationale for WTSHI and WHSI, developed in the remainder of this chapter.

---

## 2.2 Hate Speech Detection Models

The automatic detection of hate speech has evolved substantially over the past decade, progressing from rule-based lexical approaches toward increasingly sophisticated deep learning architectures. Early work by Davidson et al. (2017) laid the conceptual groundwork by demonstrating that hate speech and offensive language are empirically distinct categories — a distinction that has proven foundational to subsequent classification design. Their three-class framework (hate, offensive, neither) established that treating these categories as interchangeable inflates false-positive rates, a problem that Fortuna and Nunes (2018) later formalized in their comprehensive survey of the field, noting that inconsistent labelling conventions across studies rendered cross-system comparisons largely unreliable. These tensions around definition and operationalisation remain unresolved in the literature and bear directly on the classification choices made in this project.

The emergence of transformer-based models fundamentally shifted the performance ceiling for hate speech detection. Mozafari, Farahbakhsh, and Crespi (2020) demonstrated that fine-tuning BERT on hate speech corpora substantially outperformed prior CNN and LSTM baselines, establishing transfer learning as the de facto approach. Caselli et al. (2021) extended this line with HateBERT, a domain-adapted BERT variant retrained on abusive language corpora, showing that domain-specific pretraining yields further gains over general-purpose transformers. Systematic reviews by Jahan and Oussalah (2023) and Rawat, Kumar, and Samant (2024) collectively surveying over 130 studies confirm that transformer architectures have become the dominant paradigm since 2021, with Roy et al. (2020) representing the CNN-based generation that transformers have largely superseded. More recent work by Piot, Martín-Rodilla, and Parapar (2024), drawing on a meta-collection of 36 datasets comprising 1.2 million samples, and the transformer-era systematic review by Ramos et al. (2024) in *Social Network Analysis and Mining*, synthesising over 100 studies, consolidate this trajectory while identifying that cross-domain generalisation — that is, performance degradation when a model trained on one platform is applied to another — remains an open problem.

The emerging role of large language models has introduced a further architectural fork. Pan, García-Díaz, and Valencia-García (2024) conducted an empirical comparison of fine-tuning, zero-shot, and few-shot strategies with LLMs in English hate speech detection, finding that fine-tuned encoders retained an advantage on well-defined taxonomies. This finding is reinforced by Ghorbanpour et al. (2025), who evaluated LLaMA, Aya, and Qwen across eight languages. Jahan, Oussalah, Beddia, Mim, and Arhab (2024) situate augmentation as a complementary strategy across legacy ML, BERT, and LLM paradigms. Collectively, this body of work establishes fine-tuned encoder models as the appropriate choice for domain-constrained classification tasks — the design rationale underlying this project's use of an EDOS-trained TF-IDF logistic regression model blended with Detoxify.

Women-specific and misogyny-oriented detection introduces additional complexity. Pamungkas, Basile, and Patti (2020) demonstrated in a multilingual, cross-domain study on Twitter that misogyny classifiers trained on one platform exhibit significant performance degradation when applied to another, even within the same language — a cross-domain transfer problem that applies directly to this project's multi-platform design. Parikh et al. (2021) further showed that coarse binary sexism labels obscure substantive taxonomic variation, arguing for fine-grained categorisation across verbal abuse, objectification, and threat sub-types. Liu, Burnap, Alorainy, and Williams (2019) specifically applied fuzzy logic to text classification for ambiguous hate speech instances, demonstrating that linguistic uncertainty in harm categories is better handled through graduated membership than hard classification — a result that directly motivates this project's Mamdani fuzzy inference engine. Toxicity detection tools such as Perspective API — whose updated multilingual character-level architecture is documented by Lees et al. (2022) — serve in this literature as ensemble components rather than standalone classifiers. Finally, Alkomah and Ma's (2022) review provides additional taxonomic grounding for the four-dimensional (T, Th, F, N) construct used in WHSI.

**Research gap identified.** Despite this extensive model development, a consistent limitation across the literature is that existing classifiers operate at the comment level and are validated on the same distribution on which they were trained. No study reviewed here addresses the problem of aggregating comment-level outputs into a platform-level severity index, nor does any work formally separate the speaker role — perpetrator attack versus victim disclosure — as a pre-classification step. The result is that classifiers applied to platforms where women discuss harm in support spaces (such as Reddit's TwoXChromosomes) systematically inflate platform harm estimates by treating victim narration as equivalent to perpetrator attack. This project's WTSHI construct directly addresses this gap, but the classifier component remains bounded by a performance ceiling that the field has not yet resolved for cross-domain, multi-platform settings (EDOS holdout F1 ≈ 0.58). Human validation on the live corpus is therefore a necessary part of establishing WTSHI, not an optional extra.

---

## 2.3 Datasets and Benchmarks

The development of annotated hate speech datasets has closely tracked the theoretical debates in detection modelling, with early corpora establishing foundational taxonomies and later benchmarks introducing progressively richer annotation schemes. Waseem and Hovy (2016) produced one of the earliest structured datasets, labelling approximately 16,000 tweets for racism and sexism, establishing Twitter as the dominant collection platform for a subsequent decade. Davidson et al.'s (2017) 24,783-tweet corpus, crowdsourced via CrowdFlower, extended this to a three-class scheme and remains the most widely cited hate speech dataset; this project's Twitter supplementary panel uses the Davidson corpus directly as an upper-bound anchor. Founta et al. (2018), compiling 80,000 labelled tweets across four categories, demonstrated that large-scale crowdsourcing substantially improves inter-annotator reliability over small expert panels, a finding that shaped subsequent benchmark construction practice.

The trajectory from general hate speech to women-specific benchmarks represents a particularly relevant strand for this project. Basile et al. (2019) produced the first large-scale multilingual benchmark specifically targeting hate speech against women and immigrants through SemEval-2019 Task 5, covering 13,000 tweets in English and Spanish. Guest et al. (2021) advanced this with 6,567 expert-annotated Reddit posts structured around a hierarchical misogyny taxonomy — a corpus whose platform source directly parallels this project's Reddit live scrape. The most directly relevant benchmark is Kirk et al.'s (2023) SemEval-2023 Task 10 EDOS dataset: 20,000 fine-grained labelled social media comments (Reddit and Gab) with a hierarchical sexism taxonomy spanning five categories and eleven sub-categories. EDOS serves as the training and holdout anchor for this project's classifier, and its fine-grained label structure underlies the WTSHI perpetrator/victim distinction. Mathew et al.'s (2021) HateXplain extends the annotation space further still, introducing rationale spans alongside target community labels across 20,148 posts from Twitter and Gab, enabling explainability-oriented evaluation that binary and three-class benchmarks cannot support.

Toxicity-oriented benchmarks complement the hate speech corpora by providing continuous severity scores rather than categorical labels. The Jigsaw Toxic Comment Classification Challenge dataset (2018), comprising approximately 230,000 Wikipedia comments labelled across six toxicity sub-types, forms the training foundation of the Perspective API used in this project's toxicity scoring layer. The Jigsaw Unintended Bias dataset (2019), extending this to two million samples with fractional toxicity labels and identity annotations, introduced the problem of systematic classifier bias against identity-marked communities — a concern directly relevant to this project's multi-platform aggregation, where differential classifier behaviour across platforms could introduce non-comparability in WHSI scores. Kennedy et al. (2020) formalised the case for continuous severity measurement, proposing a faceted Rasch measurement approach applied to a hate speech corpus that yields interval-scaled severity scores rather than ordinal labels — the conceptual basis for WHSI's continuous 0–100 scoring design.

Functional evaluation benchmarks represent a more recent development. Röttger et al. (2021) introduced HateCheck, a suite of 29 functional tests covering 3,728 cases that systematically probe model failures on negation, reclamation, and target-group variation — capabilities that binary accuracy on holdout sets cannot reveal. Their follow-up multilingual extension, mHateCheck (2022), broadened coverage to eight languages. These functional evaluation tools are particularly relevant for any project that relies on a classifier trained on one distribution (EDOS) and applies it to a different one (live YouTube, Reddit, and Telegram scrapes), since cross-distribution failures are precisely what functional tests are designed to surface.

Several cross-dataset and meta-analytic studies frame the broader validity challenges. Fortuna, Soler, and Wanner (2020) analysed six major hate speech datasets and found substantial definitional inconsistency across what is labelled as toxic, hateful, offensive, or abusive, concluding that models trained on different datasets are often measuring different constructs — a finding with direct implications for this project's use of EDOS labels applied to non-EDOS content. Poletto et al.'s (2021) systematic review of benchmark corpora in *Language Resources and Evaluation* documents the variation in annotation guidelines, inter-annotator agreement reporting, and platform sampling strategies that make cross-dataset comparison unreliable. Alkomah and Ma (2022) similarly note that the dominance of Twitter as a collection platform in pre-2022 datasets creates a systematic coverage gap for Reddit, Telegram, and other platform types.

On the moderation measurement side, Chandrasekharan et al. (2017) provided one of the few empirically grounded studies of platform moderation efficacy, finding that Reddit's 2015 subreddit bans measurably reduced hate speech spillover to non-banned communities — the evidential basis for this project's moderation bypass rate (MBR) estimate for Reddit. Ribeiro, Cheng, and West (2022/23) demonstrated through a large-scale causal study that automated content moderation increases community guideline adherence, providing empirical grounding for this project's proactive detection rate (PDR) sub-component of MRI. Trujillo, Fagni, and Cresci (2023/25) audited platform-reported self-moderation data in the DSA Transparency Database at scale, finding systematic inconsistency — a finding that directly contextualises this project's decision to blend transparency-report MRI inputs with empirical removal-flag observations rather than accepting transparency data at face value. Dubois and Reepschlager (2024) traced harassment and hate speech policy evolution across Facebook, Twitter, and Reddit from 2005 to 2020, situating the MRI regulatory accountability sub-metric (RAS) in a longer longitudinal policy context. Röttger et al.'s (2022) analysis of prescriptive versus descriptive annotation paradigms in subjective NLP tasks informs the annotation design choices in this project's live-corpus validation sample.

**Research gap identified.** Two interconnected gaps are evident in this literature. First, there is no established methodology for aggregating comment-level annotations into a platform-level comparative index. Arora et al. (2023), in a gap analysis of 500 platform-facing harmful content papers, explicitly note that academic detection work focuses on single-post classification while platforms require system-level accountability metrics — precisely the gap this project's dual-index design addresses. Second, no existing benchmark separates speaker role (perpetrator versus victim) as a structural feature of the annotation schema: datasets in this literature assign labels to content, not to the discourse position of the speaker. Fortuna and Nunes (2018) acknowledge this as a known limitation without proposing a solution; Kennedy et al. (2020) introduce severity gradation but retain content-level labelling. The absence of a role-aware benchmark means that any model trained on existing corpora will conflate victim disclosure in support spaces with perpetrator attack — the specific validity problem this project's WTSHI construct is designed to correct, and one that corpus-specific human validation is necessary to formally establish.

---

## 2.4 Platform-Level Analysis

Social media has proven to be a quick vector for spreading harmful content, either directly through posts, or indirectly through comments and resharing across platforms. Platform-level analysis matters because platforms like YouTube and Instagram are far more heavily moderated compared to platforms like Reddit, Gab, or 4chan. Because of this difference, it becomes important to understand how effective moderation actually is, whether users are satisfied with it, and whether these platforms can be made into safe spaces for discussion without harming anyone. However, most existing research treats this as a comment-level problem, evaluating individual pieces of content rather than examining platforms as a whole. This review focuses on drawing comparisons between different platforms, the different ways harmful content spreads, the models developed to tackle this issue, and how effective they have been.

### Predicting and tracing how hate spreads

Content posted on different social media platforms tends to have a different effect on people, and on the discussions that they have. A study on hate comments on Twitter concluded that in order to classify a post as hateful, a model should not just take into account the words of the post, but rather the context, and whether that context can lead to comments that are harmful. Meng, Suresh, Lee, and Chakraborty propose DRAGNET++ as a new and improved model to its predecessor, DRAGNET, tested on three datasets. It combines both labeling a post as hateful and creating a tree using a GNN to see how the post can propagate hateful comments, and based on both of these signals, a decision is made on whether the post should be removed. Hence, it is important to have the structure of a conversation for predicting how hate speech spreads and intensifies.

While DRAGNET++ shows that hate can be predicted and traced within a single platform, an equally important question is whether moderating a post on one platform is enough to stop it from spreading elsewhere. While moderation on some platforms like YouTube is effective (with a lot of delays though), a study on the spread of moderated YouTube videos through platforms like Twitter (La Gatta, Luceri, Fabbri, and Ferrara) has proven that individual platform moderation does not ensure that content remains unseen by many. It was also recognized that mobilizers of these videos only interact with other mobilizers, and it was also concluded that in most circumstances, the videos become viral on a platform like Twitter and remain viral, while only being moderated on YouTube much later — thus leading us to conclude that individual moderation on one platform is not enough, and that moderation should be cross-platform, since that is how harmful and hateful content is seen to spread.

### Platform structure shapes discourse culture

Beyond how hate spreads across platforms, it is also important to examine how the structure and culture of a platform shapes the character of hate itself, based on the level of moderation offered. When Reddit and 4chan were compared during the 2020 US Elections (Zahrah, Nurse, and Goldsmith) on the basis of when people posted, who posted, what topics they discussed, which emotions they expressed, and what files they shared, it was recognized that discussions on Reddit were more organized and rooted in opinions, while on 4chan emotions ran raw and strong as a means of expressing strong beliefs. This pattern of platform design shaping discourse is not limited to election-related content; it also extends to how users discuss geopolitical conflict. While comparing discussion on geopolitical tension (the India-Qatar issue and the India-Pakistan crisis) on both Twitter and Reddit, Vasist, Krishnan, and Agnihotri made certain observations. Using social network and sentiment analysis, it was found that Twitter's profile-based structure promotes rapid information diffusion, polarization, and anger-driven discourse, whereas Reddit's community-based structure encourages more diverse viewpoints, trust, and deliberative discussion. Thus, we can conclude that platform design significantly shapes how geopolitical information spreads and how public opinion forms online.

### Improving detection models and benchmarks

Having established how hate spreads and how platform structure shapes it, the next question is how well current detection models actually perform and how they can be improved. The power that media has to influence the emotions of users should also be taken into consideration. Kim, Kim, and Kim developed a hate speech detection model for online news comments that considers not only the comment itself but also the topic and emotional framing of the news article. Using Agenda-Setting Theory, the model analyzes how media influences reader reactions and combines topic matching with emotion detection to improve hate speech classification. The approach outperformed traditional machine learning, deep learning, and transformer-based baselines, showing that article context is crucial for detecting subtle and evolving hate speech.

A related concern is not just whether a model can detect hate, but whether it can explain its decision to the user. While models can successfully conclude whether a comment is sexist or not, is that really enough for us to moderate a post or comment without a proper explanation to the user? There are many instances where people are reported on social media platforms without any reasoning, and sometimes even with the wrong conclusion. To prevent this, it is better to have models that can not only say whether a post is sexist or not, but also classify what type of sexism it is, offering an explanation to the user while at the same time ensuring that wrong reports do not happen. A competition was held in 2023 (Kirk et al., SemEval-2023 Task 10 / EDOS) where models had to be developed to (a) predict whether a comment is sexist or not, and (b) categorize it into threats, derogatory remarks, animosity, etc. Models achieved an accuracy of only about 56% on task (b). Thus, while models find it difficult to categorize comments with precision, it remains important to develop models that can do so with better accuracy.

Finally, beyond what a model detects, the choice of model architecture itself plays a major role in performance. Philipo, Sarwatt, Ding, Daneshmand, and Ning compare five transformer-based models — BERT, RoBERTa, XLNet, DistilBERT, and GPT-2 — for cyberbullying detection across multiple social media datasets. RoBERTa achieved the highest classification accuracy, DistilBERT provided the best computational efficiency, and BERT offered the best balance between performance and resource usage. The study concludes that transformer encoders are significantly more effective than generative models like GPT-2 for cyberbullying detection tasks.

Taken together, these studies reveal three important but disconnected threads of research. Work like DRAGNET++ and the YouTube-Twitter study show that hate must be understood as something that propagates — both within a single conversation and across platform boundaries — rather than as an isolated, static post. Studies comparing Reddit, 4chan, Twitter, and geopolitical discourse demonstrate that platform structure and moderation philosophy directly shape the character of harmful content, meaning platforms cannot be evaluated using a single, uniform standard. Finally, work on agenda-setting, explainable sexism classification, and transformer model comparison shows that detection itself is still an evolving and imperfect science, particularly when it comes to fine-grained categorization and contextual understanding.

However, no study reviewed here attempts to combine these threads into a single, comparative measure. Each paper either focuses on a single platform, a single model, or a single dimension of harm, but none produce an explainable platform-level score focused specifically on women-targeted harm. Furthermore, none of these works connect content-level harm with platform response, that is, none ask whether the platforms most prone to producing or spreading harmful content are also the ones doing the least to address it. This is precisely the gap the proposed Women Harassment Severity Index (WHSI) is designed to close on the harm side: a unified, cross-platform, explainable score that draws on the detection and propagation techniques shown here. Whether those scores can be read as a strict ranking is an empirical question for later chapters, not a claim the literature already supports. The accountability half of the same gap is taken up by MRI in Section 2.5.

---

## 2.5 Moderation and Accountability

While Section 2.4 examined the dynamics of harmful content and its spread across platforms, knowing what harm is, is only half the story. The platforms say they moderate this content, but how reliably they report on that moderation, how fast it actually happens and whether the systems that make these decisions can be trusted at all, are open questions. Here we move from the content itself to the accountability structures that are meant to address it. We review research on three related concerns: how well platforms document their own moderation practices, whether moderation speed is in fact fast enough to reduce harm, and how transparent and consistent the artificial intelligence systems that are doing this moderation are.

### Moderation auditing self-report

The EU has created the Digital Services Act Transparency Database (DSA-TDB) to enhance accountability, requiring platforms to disclose information regarding each moderation decision, including the nature of the content flagged, the time of the action, and whether the decision was made automatically or by a human. Analyzing the first 353 million records submitted to this database, Trujillo, Fagni, and Cresci uncovered significant inconsistencies in how platforms reported moderation actions, automation levels, content categories, and enforcement decisions. Cross-referencing with public transparency reports revealed especially large discrepancies for X, TikTok, and Meta. The research concluded that the DSA-TDB is a positive step toward accountability but the self-reported moderation data should be treated with care and validated against external sources rather than taken at face value.

A follow-up audit by Shahi, Tessa, Trujillo, and Cresci tested how reliable this data really was in real-world conditions, looking at 1.58 billion moderation actions across eight major platforms during the 2024 European Parliament elections. The researchers wanted to see whether platforms changed their moderation behaviour during a period of heightened democratic risk. Surprisingly, moderation activity did not increase much, and there was no clear improvement in how quickly content was handled. In some cases, election-related content was only moderated weeks after it was posted, suggesting that platforms were often reacting to harmful content rather than preventing it early. The reporting problems found in the first audit also continued. Some platforms recorded impossible timestamps, suggesting that content had been moderated before it was even posted, while 41% of all actions were placed under the vague category “scope of platform service.” X also stood out, reporting only 628,000 actions with almost no recorded delays and claiming that most of its moderation was done manually. Overall, the authors found that although the DSA-TDB is a useful initiative, it is not yet reliable enough to be used on its own to track platform behaviour during sensitive events. Instead, it needs to be combined with external datasets and other sources of evidence.

Taken together, the two audits show just how difficult transparency and accountability in content moderation still are. Even when platforms are legally required to report their moderation activity, the data they provide can be inconsistent, incomplete, and difficult to interpret. This ultimately raises a bigger question: how much can researchers and the public really trust platforms to accurately report what they are doing to keep users safe?

### Does moderating speed truly lessen harm?

If platforms’ internal moderation data isn’t always reliable, the larger issue is whether moderation occurs swiftly enough to have a real impact. Schneider and Rizoiu looked at how harmful content propagates by applying Hawkes processes, a mathematical model designed to track viral cascades. The researchers concentrated on two metrics: potential harm, which assesses how widely a post can spread, and content half-life, which indicates how quickly a post garners most of its engagement. Analyzing Twitter data from 2022, they discovered that the EU’s 24-hour takedown policy was significantly more effective for content that spread slowly, like `#climatescam`, cutting potential harm by 29%. For content that spreads quickly, such as `#americafirst`, the reduction was only 13%, because most of the engagement had already occurred within those 24 hours. Interestingly, moderation proved especially effective for highly viral content, as deleting a single viral post could also prevent numerous reposts and interactions. This implies that, with limited moderation resources, platforms might need to prioritize content spreading most rapidly rather than applying equal treatment to all posts.

In a second study, Truong, Kim, Nogara, and colleagues tested this connection between speed and real-world exposure directly using DSA-TDB data combined with simulation (SimSoM) to assess whether eliminating illegal content leads to fewer users encountering it. Average delays in content removal differed significantly by platform, ranging from six days on TikTok to 87 days on Instagram and 286 days on YouTube. The study found that content taken down within nine hours reduced harmful exposure by 95–100%, but if it stayed online for days or weeks, most users had already been exposed, regardless of later removal. The authors concluded that takedown speeds on most major platforms are currently too sluggish to effectively reduce exposure, but they also warn that excessively tight deadlines could lead to excessive removals and censorship. Together, these two studies show that moderation speed is not just a formality but the most critical factor in determining whether moderation is effective and that most platforms, in practice, are too slow for it to make any real difference.

### Can AI moderation be trusted?

Even when moderation occurs swiftly, a key issue remains: whether the AI systems carrying it out are transparent and consistent enough to earn trust. Zangl, Loi, Zachos, and colleagues examined toxicity detection tools on civic engagement platforms, which tend to disproportionately deter participation from minority groups. Under the EU AI Act, these systems are categorized as high-risk, but widely adopted tools such as Perspective API and TrollWall AI remain largely opaque. Current explainability techniques like LIME and SHAP lack a unified standard for auditing bias or fairness. The authors contend that moderation systems ought to disclose precisely which words or signals led to a decision, instead of delivering vague or unexplained rulings. This worry about lack of transparency is made worse by a more fundamental issue: inconsistency. Gomez, Machado, Paes, and Calmon trained the same moderation model repeatedly with different random seeds and discovered that even when the models performed similarly overall, they often reached different decisions on individual posts — about 30–34% of posts received varying moderation outcomes solely based on the random seed, with disagreement climbing to 35% for anti-LGBTQ content. In some instances, the models contradicted one another, despite all human annotators having reached a unanimous agreement on the correct label.

This indicates that moderation outcomes may depend on arbitrary technical decisions rather than transparent, reproducible guidelines, which raises significant concerns about fairness and accountability because high accuracy does not ensure that any individual decision is fair or consistent. Together, these two studies indicate that AI moderation tools cannot be viewed as impartial judges at this time: their decision-making processes are frequently opaque to the users they oversee, and their outputs may vary even when presented with the same input, thereby weakening the accountability they are supposed to ensure.

### Research gap in moderation and accountability

Overall, this section highlights three interrelated shortcomings in existing research on moderation and accountability. Audits of the DSA-TDB reveal that platforms cannot be relied upon to accurately self-report moderation, even when legally required to do so. Research on moderation speed indicates that even when moderation is genuinely tried, it is often too slow to stop harmful content from being viewed. Research into AI moderation systems reveals that the tools used for this purpose are frequently opaque and internally inconsistent, which means issues with speed and reporting are further worsened by a lack of trust in the decision-making process itself.

However, none of these studies integrate the three dimensions — reporting reliability, response speed, and decision consistency — into one unified metric for assessing a platform’s accountability, nor do any link this accountability framework back to the platform-level harm patterns outlined in Section 2.4. No prior research investigates whether the platforms generating or disseminating the most damaging content against women are also those that moderate such content most slowly or least reliably. This is exactly the gap that the proposed Moderation Responsiveness Index (MRI) aims to address: a standardized, interpretable metric for platform accountability, integrating findings from reporting audits, response-time assessments, and consistency issues discussed above, alongside harm-detection methods from Section 2.4, to create the unified WHSI–MRI framework.

---

## 2.6 Explainable Scoring, Fuzzy Logic, and Risk Indices

Sections 2.1–2.5 establish *what* to measure (women-targeted harm; platform response) and *why* comment-level classifiers are not enough. This section reviews *how* prior work handles uncertainty, continuous severity, and explanation — the three methodological ingredients of WHSI.

### Graduated membership rather than hard labels

Hate and sexism are linguistically ambiguous. Liu, Burnap, Alorainy, and Williams (2019) showed that fuzzy multi-task learning handles uncertain hate-type membership better than forced hard classification, which is the direct rationale for this project’s Mamdani fuzzy inference engine over the four WHSI dimensions (T, Th, F, N). The Mamdani controller itself is the classical linguistic rule base (Mamdani and Assilian, 1975), sitting on Zadeh’s (1965) fuzzy-set foundation: inputs are mapped to overlapping membership functions, fired through interpretable if–then rules, and defuzzified to a scalar. That pipeline is why WHSI can report a 0–100 score without pretending each comment is a binary “hate / not hate” atom.

Alkomah and Ma (2022) supply taxonomic support for keeping those four dimensions distinct rather than collapsing them into a single toxicity logit. Parikh et al. (2021) and Kirk et al. (2023) make the same point from the sexism side: coarse binary labels hide threat versus derogation versus animosity. Fuzzy rules over T/Th/F/N are this project’s way of retaining that granularity at aggregation time.

### Continuous severity, not ordinal buckets

Kennedy et al. (2020) argued that hate-speech measurement should yield interval-scaled severity, not ordinal class labels, using a faceted Rasch model. WHSI’s 0–100 design follows that argument: platform comparison requires a continuous construct, even if later chapters must bound how far those numbers may be compared across YouTube, Reddit, and Telegram. Jigsaw’s fractional toxicity labels (2019) and Perspective’s character-level scoring (Lees et al., 2022) are the industrial counterpart: toxicity as a score used as an *ensemble input*, not as the index itself.

### Explainability as a design constraint

Mathew et al. (2021) introduced HateXplain so that models could be judged on rationale spans, not only accuracy. Röttger et al. (2022) distinguished prescriptive from descriptive annotation — a choice this project’s validation protocol has to make explicit. Zangl et al. (Section 2.5) showed that production toxicity APIs remain opaque even where LIME and SHAP exist. WHSI therefore cannot be a black-box neural score alone: the Mamdani rule base, the WTSHI perpetrator/victim split, and the MRI sub-metrics (MBR, PDR, RAS) are the project’s answer to that explainability requirement.

### What the scoring literature does not yet do

Safety-index and risk-framework papers in adjacent domains (content-level toxicity scores; Rasch severity; fuzzy hate-type membership) still operate on posts, not platforms, and they do not isolate women as the protected population. No reviewed framework jointly (i) role-filters speech (WTSHI), (ii) fuzzifies four harm dimensions into a platform score (WHSI), and (iii) pairs that score with an accountability index (MRI). That combination is the methodological contribution this chapter is building toward.

---

## 2.7 Research Gap Summary

Across all six sections, the same absences recur:

1. **Comment-level, not platform-level.** Detection work (2.1–2.2) and benchmarks (2.3) evaluate posts. Arora et al. (2023) state the mismatch with what platforms actually need: system-level accountability metrics.
2. **Women-specific harm is modelled as a classifier task, not an index.** Section 2.1 shows strong misogyny detectors; none aggregate to a women’s platform score.
3. **Speaker role is unlabelled.** No benchmark in 2.3 separates perpetrator attack from victim disclosure. WTSHI is proposed to close that validity hole.
4. **Platforms are not comparable under one construct.** Section 2.4 shows that structure and propagation differ by platform; a uniform toxicity rate is the wrong comparative object.
5. **Harm is not paired with accountability.** Section 2.5 shows that self-report, speed, and AI consistency are measured separately, never as one MRI-style index, and never against women-targeted harm.
6. **Scoring is either binary or opaque.** Section 2.6 shows fuzzy and Rasch tools exist, but not as an explainable dual index for women.

The project therefore proposes **WHSI × MRI**, with **WTSHI** as the role-aware input to WHSI. The literature supports that design. It does not, by itself, guarantee that the resulting scores can be read as a simple cross-platform ranking — that is an empirical claim for the results chapters, bounded by classifier and scrape limits.

---

## References (Section 2.1, numbered)

[1] S. J. Prithila, F. H. Tonima, T. T. Oishi, M. N. Islam, E. R. Rhythm, A. M. Amit, and A. A. Rasel, "Detecting derogatory comments on women using transformer-based models," in *2023 IEEE International Conference on Communication, Networks and Satellite (COMNETSAT)*, 2023, pp. 1–7.

[2] E. Martinez, J. Cuadrado, J. C. Martinez-Santos, and E. Puertas, "Detection of online sexism using lexical features and transformer," in *2023 IEEE Colombian Caribbean Conference (C3)*, 2023, pp. 1–5.

[3] H. Abburi, P. Parikh, N. Chhaya, and V. Varma, "Fine-grained multi-label sexism classification using a semi-supervised multi-level neural approach," *Data Science and Engineering*, vol. 6, pp. 359–379, 2021.

[4] A. Mohasseb and E. Amer, "Enhancing misogyny detection through context-aware semantic enrichment," in *2025 20th International Workshop on Semantic and Social Media Adaptation and Personalization (SMAP)*, 2025, pp. 1–6.

[5] O. C. Akinduyite and O. Chris-Umoru, "Automatic misogyny detection using supervised artificial neural networks," *Annals of Data Science*, 2025.

[6] S. Singh, A. Haridasan, and R. Mooney, "Towards multimodal misogyny detection in memes," in *Proc. 7th Workshop on Online Abuse and Harms (WOAH)*, 2023, pp. 150–159.

[7] S. Karishma and V. Akila, "A comparative analysis of multimodal misogyny memes using deep learning with semi supervised learning algorithms," in *Proc. 3rd Int. Conf. on Self Sustainable Artificial Intelligence Systems (ICSSAS)*, 2025, pp. 1791–1796.

[8] M. Z. U. Rehman, A. Shah, and N. Kumar, "Detecting implicit sexism in digital social networks via contrastive learning-based adaptive network," *Computers and Electrical Engineering*, vol. 137, p. 111261, 2026.

[9] F. Rodriguez-Sanchez, J. Carrillo-de-Albornoz, and L. Plaza, "Leveraging unsupervised task adaptation and semi-supervised learning with semantic-enriched representations for online sexism detection," *Expert Systems*, vol. 42, no. e13763, 2024.

[10] U. S. Yadavalli and S. R. Sahoo, "A multi-granular hybrid neural architecture for detecting abusive content in online social networks (OSNs) with contextual awareness," *Journal of Big Data*, vol. 13, no. 5, 2026.

[11] P. D. Kaware and A. B. Raut, "Automatic detection of multilingual misogynistic content in social media data based on machine learning approach," in *Proc. 2024 Int. Conf. Integrated Circuits and Communication Systems (ICICACS)*, 2024, pp. 1–7, doi: 10.1109/ICICACS60521.2024.10499136.

[12] E. Altarawneh, K. Pokhrel, D. Chandola, G. Melo, and K. Soldatic, "Beyond monolithic LLMs: Modular AI for online harassment detection," in *Proc. 2025 IEEE Int. Conf. Collaborative Advances in Software and Computing (CASCON)*, 2025, pp. 1–10, doi: 10.1109/CASCON66301.2025.00021.

---

## References (Sections 2.2–2.6, author–year)

Alkomah, F., & Ma, X. (2022). A literature review of textual hate speech detection methods and datasets. *Information, 13*(6), 273.

Arora, A., Nakov, P., Hardalov, M., Sarwar, S. M., Nayak, V., Dinkov, Y., Zlatkova, D., Dent, K., Bhatawdekar, A., Bouchard, G., & Augenstein, I. (2023). Detecting harmful content on online platforms: What platforms need vs. where research efforts go. *ACM Computing Surveys*. https://doi.org/10.1145/3603399

Basile, V., et al. (2019). SemEval-2019 Task 5: Multilingual detection of hate speech against immigrants and women in Twitter. *SemEval*. https://aclanthology.org/S19-2007/

Caselli, T., Basile, V., Mitrović, J., & Granitzer, M. (2021). HateBERT: Retraining BERT for abusive language detection in English. *WOAH 2021*, 17–25. https://aclanthology.org/2021.woah-1.3/

Chandrasekharan, E., et al. (2017). You can’t stay here: The efficacy of Reddit’s 2015 ban examined through hate speech. *PACM HCI (CSCW)*. https://doi.org/10.1145/3134666

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language. *ICWSM, 11*(1), 512–515. https://doi.org/10.1609/icwsm.v11i1.14955

Dubois, E., & Reepschlager, A. (2024). How harassment and hate speech policies have changed over time: Comparing Facebook, Twitter and Reddit (2005–2020). *Policy & Internet, 16*(3), 523–542. https://doi.org/10.1002/poi3.387

Fortuna, P., & Nunes, S. (2018). A survey on automatic detection of hate speech in text. *ACM Computing Surveys, 51*(4), 85. https://doi.org/10.1145/3232676

Fortuna, P., Soler, J., & Wanner, L. (2020). Toxic, hateful, offensive or abusive? What are we really classifying? *LREC*, 6786–6794. https://aclanthology.org/2020.lrec-1.838/

Founta, A., et al. (2018). Large scale crowdsourcing and characterization of Twitter abusive behavior. *ICWSM*. https://doi.org/10.1609/icwsm.v12i1.14991

Ghorbanpour, F., Dementieva, D., & Fraser, A. (2025). Can prompting LLMs unlock hate speech detection across languages? *WOAH 2025*. https://aclanthology.org/2025.woah-1.39/

Gomez, J. F., Machado, C. V., Paes, L. M., & Calmon, F. P. (2024). Algorithmic arbitrariness in content moderation. *FAccT*. https://doi.org/10.1145/3630106.3659036

Guest, E., Vidgen, B., Mittos, A., Sastry, N., Tyson, G., & Margetts, H. (2021). An expert annotated dataset for the detection of online misogyny. *EACL*, 1336–1350. https://aclanthology.org/2021.eacl-main.114/

Jahan, M. S., & Oussalah, M. (2023). A systematic review of hate speech automatic detection using NLP. *Neurocomputing, 546*, 126232.

Jahan, M. S., Oussalah, M., Beddia, D. R., Mim, J. K., & Arhab, N. (2024). A comprehensive study on NLP data augmentation for hate speech detection: Legacy methods, BERT, and LLMs. arXiv:2404.00303.

Kennedy, C. J., et al. (2020). Constructing interval-valued hate speech scores with faceted Rasch measurement. arXiv:2009.10277.

Kim, S.-S., Kim, S., & Kim, H.-W. (2025). Hate speech detection on online news platforms: A deep-learning approach based on agenda-setting theory. *Journal of Management Information Systems, 42*(3), 673–705.

Kirk, H. R., Yin, W., Vidgen, B., & Röttger, P. (2023). SemEval-2023 Task 10: Explainable Detection of Online Sexism (EDOS). *SemEval*, 2193–2210. https://aclanthology.org/2023.semeval-1.305/

La Gatta, V., Luceri, L., Fabbri, F., & Ferrara, E. (2023). The interconnected nature of online harm and moderation. *WebSci*. https://doi.org/10.1145/3603163.3609058

Lees, A., Tran, V. Q., Tay, Y., Sorensen, J., Gupta, J., Metzler, D., & Vasserman, L. (2022). A new generation of Perspective API: Efficient multilingual character-level transformers. *KDD*. https://doi.org/10.1145/3534678.3539147

Liu, H., Burnap, P., Alorainy, W., & Williams, M. L. (2019). Fuzzy multi-task learning for hate speech type identification. *WWW ’19*, 3006–3012. https://doi.org/10.1145/3308558.3313546

Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies, 7*(1), 1–13.

Mathew, B., et al. (2021). HateXplain: A benchmark dataset for explainable hate speech detection. *AAAI, 35*(17), 14867–14875.

Meng, Q., Suresh, T., Lee, R. K.-W., & Chakraborty, T. (2022). Predicting hate intensity of Twitter conversation threads. arXiv:2206.08406. (DRAGNET++)

Mozafari, M., Farahbakhsh, R., & Crespi, N. (2020). Hate speech detection and racial bias mitigation in social media based on BERT model. *PLoS ONE, 15*(8), e0237861.

Pamungkas, E. W., Basile, V., & Patti, V. (2020). Misogyny detection in Twitter: A multilingual and cross-domain study. *Information Processing & Management, 57*(6), 102360.

Pan, R., García-Díaz, J. A., & Valencia-García, R. (2024). Comparing fine-tuning, zero and few-shot strategies with LLMs in hate speech detection in English. *CMES, 140*(3), 2849–2868.

Parikh, P., Abburi, H., Chhaya, N., Gupta, M., & Varma, V. (2021). Categorizing sexism and misogyny through neural approaches. *ACM Transactions on the Web, 15*(4).

Philipo, A. G., Sarwatt, D. S., Ding, J., Daneshmand, M., & Ning, H. (2025). Assessing text classification methods for cyberbullying detection on social media platforms. *IEEE TIFS*. arXiv:2412.19928.

Piot, P., Martín-Rodilla, P., & Parapar, J. (2024). MetaHate: A dataset for unifying efforts on hate speech detection. *ICWSM, 18*, 2025–2039.

Poletto, F., Basile, V., Sanguinetti, M., Bosco, C., & Patti, V. (2021). Resources and benchmark corpora for hate speech detection: A systematic review. *Language Resources and Evaluation, 55*, 477–523.

Ramos, G., et al. (2024). A comprehensive review on automatic hate speech detection in the age of the transformer. *Social Network Analysis and Mining, 14*, 204.

Rawat, A., Kumar, S., & Samant, S. S. (2024). Hate speech detection in social media: Techniques, recent trends, and future challenges. *WIREs Computational Statistics, 16*(2), e1648.

Ribeiro, M. H., Cheng, J., & West, R. (2022). Automated content moderation increases adherence to community guidelines. arXiv:2210.10454.

Roy, P. K., Tripathy, A. K., Das, T. K., & Gao, X.-Z. (2020). A framework for hate speech detection using deep convolutional neural network. *IEEE Access, 8*, 204951–204962.

Röttger, P., et al. (2021). HateCheck: Functional tests for hate speech detection models. *ACL-IJCNLP*, 41–58. https://aclanthology.org/2021.acl-long.4/

Röttger, P., Vidgen, B., Hovy, D., & Pierrehumbert, J. B. (2022). Two contrasting data annotation paradigms for subjective NLP tasks. *NAACL*, 175–190. https://aclanthology.org/2022.naacl-main.13/

Schneider, P. J., & Rizoiu, M.-A. (2023). The effectiveness of moderating harmful online content. *PNAS, 120*(34), e2307360120.

Shahi, G. K., Tessa, B., Trujillo, A., & Cresci, S. (2025). A year of the DSA Transparency Database: What it (does not) reveal about platform moderation during the 2024 European Parliament election. arXiv:2504.06976.

Trujillo, A., Fagni, T., & Cresci, S. (2025). The DSA Transparency Database: Auditing self-reported moderation actions by social media. *PACM HCI*. https://doi.org/10.1145/3711085 (arXiv:2312.10269, 2023)

Truong, B. T., Kim, S., Nogara, G., Verdolotti, E., Sahneh, E. S., Saurwein, F., Just, N., Luceri, L., Giordano, S., & Menczer, F. (2025). Delayed takedown of illegal content on social media makes moderation ineffective. arXiv:2502.08841.

Vasist, P. N., Krishnan, S., & Agnihotri, P. (2025). The unfolding of geopolitical tensions on social networks: A social network analysis of Twitter and Reddit conversations. *Internet Research*. https://doi.org/10.1108/intr-02-2024-0155

Waseem, Z., & Hovy, D. (2016). Hateful symbols or hateful people? *NAACL SRW*, 88–93. https://aclanthology.org/N16-2013/

Zadeh, L. A. (1965). Fuzzy sets. *Information and Control, 8*(3), 338–353.

Zahrah, F., Nurse, J. R. C., & Goldsmith, M. (2022). A comparison of online hate on Reddit and 4chan: A case study of the 2020 US election. *ACM SAC*. arXiv:2202.01302.

Zangl, M., Loi, I., Zachos, P., Bedek, M., Dimogerontakis, E., Nikolaou, C.-E., Albert, D., & Moustakas, K. (2025). A multidisciplinary analysis of transparent AI-driven toxicity detection tools for civic engagement platforms. *AI & Society*. https://doi.org/10.1007/s00146-025-02424-5
