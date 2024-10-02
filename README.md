# Fairness in LLM
This repository contains a collection of papers, tools, datasets for fairness of LLM.

- [Introduction](#introduction)
  - [Sources of Bias in LLMs](#sources-of-bias-in-llms)
  - [Assessing Fairness in LLMs](#assessing-fairness-in-llms)
  - [Mitigation Strategies](#mitigation-strategies)
  - [Challenges and Future Directions](#challenges-and-future-directions)
- [Metrics](#metrics)
- [Tools](#tools)
  - [LLM tools](#llm-tools)
  - [General tools](#general-tools)
  - [Explanation tools](#explanation-tools)
- [Datasets](#datasets)
- [Papers](#papers)
  - [Survey papers](#survey-papers)
  - [General papers](#general-papers)  
- [Other Useful Links](#other-useful-links)


# Introduction
Fairness in Large Language Models (LLMs) is a critical area of research, as these models are increasingly used in applications that impact diverse populations. The challenge lies in ensuring that LLMs do not perpetuate or exacerbate biases, particularly against marginalized groups. This involves understanding the sources of bias, developing metrics for assessing fairness, and implementing strategies to mitigate unfair outcomes. The following sections explore these aspects in detail, drawing insights from recent research.

## Sources of Bias in LLMs
- Data and Training Bias: LLMs are trained on vast datasets that may contain historical biases, leading to biased model outputs. These biases can manifest in various forms, such as gender, race, and age discrimination.
- Model Architecture and Design Bias: The design of LLMs can inherently favor certain types of data or interactions, which may not be representative of all user groups.
- Use Case Specific Bias: Different applications of LLMs, such as text classification or information retrieval, can introduce unique biases based on how the models are used and the context in which they operate.

## Assessing Fairness in LLMs
- Fairness Metrics: Various metrics have been proposed to evaluate fairness in LLMs, including counterfactual metrics and stereotype classifiers. These metrics help in quantifying bias and assessing the fairness of model outputs. 
- Individual vs Group Fairness Evaluation: Evaluating LLMs through a group fairness lens involves analyzing how different social groups are represented and treated by the models. This approach uses datasets like GFair to assess biases across multiple dimensions.
- Empirical Studies: Studies using datasets such as the TREC Fair Ranking dataset provide empirical benchmarks for evaluating fairness in LLMs, particularly in ranking tasks.

## Mitigation Strategies
Bias mitigation techniques in LLMs are organized based on the stage of the model’s lifecycle at which they are applied: pre-processing, in-training, intra-processing, and post-processing. Pre-processing techniques involve modifying the training data to reduce biases before they are learned by the model, such as by aug-menting data or adjusting data sampling methods. In-training techniques modify the learning algorithm itself, including adjustments to the loss function or model architec-ture to counteract biased learning patterns. Intra-processing methods focus on altering the model's behavior during inference, such as by adjusting decision thresholds or re-ranking outputs. Finally, post-processing techniques involve modifying the outputs af-ter generation, typically through rules or additional models that adjust or filter the text to remove or reduce biased content. Each category of techniques offers different strat-egies to reduce bias, reflecting varying degrees of integration with the model architec-ture and training process. 
- Prompt Engineering and In-Context Learning: Techniques such as prompt engineering and in-context learning have been shown to reduce biases in LLM outputs by guiding the models to generate more equitable content.
- Remediation Techniques: Various strategies, including data resampling, model regularization, and post-processing techniques, have been proposed to mitigate biases in LLM-based applications.
- Fairness-Aware Frameworks: Frameworks that incorporate fairness regulations and definitions into the model training and evaluation process can help ensure more inclusive and fair outcomes.

## Challenges and Future Directions
- Complexity of Fairness in LLMs: The inherent complexity of LLMs and their diverse applications make it challenging to apply traditional fairness frameworks. This necessitates the development of new guidelines and iterative processes involving stakeholders to achieve fairness in specific use cases.
- Scalable Solutions: As LLMs continue to evolve, scalable solutions that leverage AI's general-purpose capabilities to address fairness challenges are needed.
- Fairness in Serving LLMs: Ensuring fairness in LLM inference services involves addressing challenges related to request scheduling and resource utilization, with novel algorithms like the Virtual Token Counter (VTC) showing promise in this area.

While significant progress has been made in understanding and addressing fairness in LLMs, challenges remain. The complexity of human-AI interactions and the diverse contexts in which LLMs are deployed require ongoing research and innovation. Moreover, achieving fairness is not just a technical challenge but also a social one, necessitating collaboration among developers, users, and stakeholders to create AI systems that are equitable and just.

# Metrics
A comprehensive list of **fairness metrics** for LLMs, organized into three main groups: **embeddings-based metrics**, **probability-based metrics**, and **generation-based metrics**.
| **Metric Type**             | **Description**                                                                                                         | **comments**                                                                                                                                                                             |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Embedding-Based Metrics** | Metrics that evaluate bias using hidden vector representations of words or sentences. They analyze how embeddings relate to protected attributes and measure the extent of bias encoded in these representations. |                                                                                                                            |
| **Probability-Based Metrics** | Metrics that assess bias based on model-assigned probabilities for tokens or outputs. These metrics analyze the likelihood of generating certain outputs for different demographic groups. |                                                                                                                                    |
| **Generation-Based Metrics** | Metrics that evaluate the fairness of the text generated by LLMs. They often involve assessing the content for stereotypes, harmful language, or biased implications in the generated text. |                                                                                                                             |


### **Embeddings-Based Fairness Metrics**

These metrics assess bias present in word or sentence embeddings produced by LLMs.

1. **Word Embedding Association Test (WEAT)**
   - Measures biases by comparing similarities between target concepts (e.g., male vs. female names) and attribute words (e.g., career vs. family).

2. **Sentence Encoder Association Test (SEAT)**
   - An extension of WEAT for sentence embeddings from models like BERT or GPT.

3. **Embedding Coherence Test**
   - Evaluates how closely embeddings of words related to specific demographics cluster together, indicating potential bias.

4. **Direct Bias Measurement**
   - Computes cosine similarity between embeddings of demographic identifiers (e.g., "man," "woman") and attribute words to quantify bias.

5. **Bias Subspace Projection**
   - Projects embeddings onto identified bias subspaces to measure the extent of bias along specific dimensions.

6. **Relative Norm Difference**
   - Assesses bias by measuring differences in the norms of embeddings associated with different demographic groups.

7. **Hardness Bias**
   - Measures how easily a classifier can distinguish between demographic groups based on embeddings; higher accuracy indicates more bias.

8. **Mahalanobis Distance Bias**
   - Uses Mahalanobis distance to detect biases in the distribution of embeddings across groups.

9. **Centering Alignment**
   - Compares the mean embeddings of different groups to assess disparities.

10. **Local Neighborhood Bias**
    - Analyzes the immediate neighbors of embeddings in the vector space for demographic clustering.


### **Probability-Based Fairness Metrics**

These metrics evaluate bias based on the probabilities or likelihoods assigned by LLMs to different sequences or tokens.

1. **Discovery of Correlations (DisCo):** DisCo assesses bias by using template sentences with two slots, such as “[PERSON] often likes to [BLANK]”. The [PERSON] slot is filled with gender-related terms, while the [BLANK] slot is completed by the model's top predictions. By analyzing how the model's predictions vary based on the gender term used, DisCo evaluates the presence and magnitude of bias in the model.


2. **Log Probability Bias Score (LPBS) :** LPBS also employs template sentences but corrects for inconsistent prior probabilities of target attributes. It computes the normalized probabilities of sentences like “he is a doctor” versus “she is a doctor” by adjusting for the baseline probabilities (e.g., “he is a [MASK]”). Bias is quantified by comparing these normalized probability scores between different demographic groups using the formula:

```math
\text{LPBS}(S) = \log\left(\frac{P_{\text{target}_i}}{P_{\text{prior}_i}}\right) - \log\left(\frac{P_{\text{target}_j}}{P_{\text{prior}_j}}\right)
```

3. **CrowS-Pairs Score:** The CrowS-Pairs Score utilizes pseudo-log-likelihood (PLL) to detect bias by analyzing pairs of sentences—one stereotypical and one neutral or counter-stereotypical. PLL approximates the probability of each token conditioned on the rest of the sentence, allowing for an overall sentence probability estimation. By comparing the PLL scores of these sentence pairs, the metric assesses the model's inclination toward stereotypes.

4. **Context Association Test (CAT):** CAT evaluates bias by comparing the probabilities assigned to a stereotype, an anti-stereotype, and a meaningless option for each sentence, focusing on \( P(M \mid U; \theta) \), where \( M \) is the meaningful option and \( U \) is the context. Unlike pseudo-log-likelihood methods that consider \( P(U \mid M; \theta) \), CAT assesses the likelihood of the model choosing the meaningful completion given the context. The bias score is calculated by averaging the log probabilities of the meaningful options across all instances.


5. **Idealized CAT (iCAT) Score:** The iCAT score refines CAT by defining an ideal language model with a language modeling score (lms) of 100 (always choosing meaningful options) and a stereotype score (ss) of 50 (choosing stereotypes and anti-stereotypes equally). It is calculated using the formula:

$$
\text{iCAT}(S) = \text{lms} \times \frac{\min(\text{ss}, 100 - \text{ss})}{50}
$$

This metric balances the model's language proficiency with its bias level, rewarding models that are both accurate and unbiased.


6. **All Unmasked Likelihood (AUL):** AUL extends metrics like CrowS-Pairs Score and CAT by evaluating the likelihood of the entire unmasked sentence, predicting all tokens without masking. This method provides the model with full context, improving prediction accuracy and reducing selection bias from masked tokens. The AUL score is the average log probability of all tokens in the sentence:

$$
\text{AUL}(S) = \frac{1}{|S|} \sum_{s \in S} \log P(s \mid S; \theta)
$$

7. **AUL with Attention Weights (AULA):** AULA enhances AUL by incorporating attention weights to account for the varying importance of tokens. Each token's log probability is weighted by its associated attention weight \( \alpha_i \), reflecting its significance in the sentence:

$$
\text{AULA}(S) = \frac{1}{|S|} \sum_{s \in S} \alpha_i \log P(s \mid S; \theta)
$$

This weighted approach provides a more nuanced bias assessment by emphasizing more influential tokens.

8. **Language Model Bias (LMB):** LMB measures bias by comparing the mean perplexity between biased statements and their counterfactuals involving alternative social groups. After removing outlier pairs with extreme perplexity values, it computes the t-value from a two-tailed Student's t-test between the perplexities of the biased and counterfactual statements. A significant t-value indicates the presence of bias in the language model's predictions.


9. **Entropy Difference**
   - Compares the uncertainty (entropy) in model predictions across groups; significant differences may signal bias.

10. **Mutual Information Bias**
    - Measures the mutual information between model outputs and sensitive attributes to detect dependency.
11. **Perplexity Disparity**
   - Compares perplexity scores for texts pertaining to different demographic groups; higher perplexity may indicate bias or unfamiliarity.

12. **Conditional Likelihood Difference**
   - Measures differences in the likelihood assigned to sequences conditioned on demographic attributes.

13. **Negative Log-Likelihood (NLL) Disparity**
   - Assesses if the model assigns higher NLL to inputs related to certain groups, indicating potential bias.
14. **Exposure Bias**
   - Evaluates the probability that the model will generate biased or harmful content when conditioned on specific inputs.

15. **Bias Amplification Metric**
   - Quantifies how much a model amplifies existing biases present in the training data during prediction.

16. **Likelihood of Stereotypical Associations**
   - Computes the probability that the model predicts stereotypical associations over neutral or unbiased ones.

17. **KL-Divergence Between Group Distributions**
   - Measures divergence in the output probability distributions across different demographic groups.

18. **Calibration Error Across Groups**
   - Assesses whether predicted probabilities are well-calibrated for different demographic groups, indicating fairness in uncertainty estimation.

### **Generation-Based Fairness Metrics**

These metrics analyze the content generated by LLMs to detect and quantify bias or unfair representations. It can be furhter classified as 

- Distribution-Based Metrics: These metrics assess the distribution of terms in the generated text. However, they may be limited as word associations with protected attributes might not accurately reflect downstream disparities.

- Classifier-Based Metrics: These involve using a classifier to evaluate the generated text for bias. However, the reliability of these metrics can be compromised if the classifier itself is biased 

- Lexicon-based metrics: These perform a word-level analysis of the generated output, comparing each word to a pre-compiled list of harmful words, or assigning each word a pre-computed bias score.

Below are some tyical metrics:

1. **Stereotype Content Generation Score**
   - Quantifies the extent to which generated text contains stereotypes or biased language against specific groups.

2. **Bias in Generated Text**
   - Measures the frequency and severity of biased or harmful language in model outputs.

3. **Toxicity Scores**
   - Uses tools like the **Perspective API** to evaluate the toxicity levels of generated text towards different demographics.

4. **Sentiment Analysis Disparity**
   - Assesses differences in sentiment expressed in generated content about different demographic groups.

5. **Demographic Representation Parity**
   - Evaluates whether different demographic groups are equally represented or mentioned in generated content.

6. **Harmful Content Detection Rate**
   - Calculates the percentage of generated outputs flagged as harmful or biased by automated detectors.

7. **Crowdsourced Bias Assessment**
   - Involves human evaluators rating the fairness or bias in generated content for qualitative insights.

8. **Diversity Metrics Across Groups**
   - Measures the diversity of generated content related to different demographic attributes to ensure varied representation.

9. **Self-BLEU Disparity**
   - Compares the diversity within generated texts for different groups by measuring BLEU scores among outputs; lower diversity may indicate bias.

10. **Average Negative Impact (ANI)**
    - Quantifies the negative impact or harm of generated content on different demographic groups.

11. **Counterfactual Fairness Evaluation**
    - Generates counterfactual instances by swapping demographic identifiers (e.g., "he" to "she") and assesses changes in the output to detect bias.

12. **Bias Sentiment Analysis**
    - Analyzes sentiment biases in generated content when referring to different groups, using sentiment analysis tools.

13. **Equity Evaluation Corpus (EEC) Metrics**
    - Uses specially designed sentences to test for bias in sentiment and toxicity towards different identities.

14. **Stereotype Score with Contextual Prompts**
    - Measures bias in generated responses when prompted with context that could elicit biased outputs.

15. **Offensiveness Scores**
    - Assesses the level of offensiveness in generated content using lexicons or classifiers.

16. **False Negative Rate Balance**
    - Ensures that the rate at which the model fails to detect harmful content is balanced across different groups.

17. **Contextualized Fairness Measures**
    - Evaluates fairness in specific contexts or scenarios relevant to different demographic groups.


These metrics help identify and quantify biases related to gender, race, religion, age, and other sensitive attributes in LLMs. Employing a combination of these metrics provides a comprehensive understanding of a model's fairness and guides efforts to mitigate biases.


# Tools
## LLM tools

| Tool                | Link                                                       | Description                                                                                                                |
|---------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| giskard              | [giskard](https://github.com/Giskard-AI/giskard)          | Open-Source Evaluation & Testing for ML models & LLMs. RAG Evaluation Toolkit (RAGET): Automatically generate evaluation datasets & evaluate RAG application answers. |
| FaiRLLM            | [FaiRLLM](https://github.com/jizhi-zhang/FaiRLLM)         | The code for "Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation".              |
| fairness-monitoring  | [fairness-monitoring](https://github.com/alan-turing-institute/fairness-monitoring) | A proactive fairness review approach in the early stages of AI development. It provides developer-oriented methods and tools to self-assess and monitor fairness. |
| wefe                | [wefe](https://github.com/dccuchile/wefe)                | The Word Embeddings Fairness Evaluation Framework. WEFE standardizes the bias measurement and mitigation in Word Embeddings models. Please feel welcome to open an issue in case you have any questions or a pull request if you want to contribute to the project! |
| SafeNLP             | [SafeNLP](https://github.com/microsoft/SafeNLP)          | Safety Score for Pre-Trained Language Models.                                                                              |
| Metric-Fairness     | [Metric-Fairness](https://github.com/txsun1997/Metric-Fairness) | Evaluation of Pre-trained language model-based metrics (PLM-based metrics, e.g., BERTScore, MoverScore, BLEURT).          |
| Dbias               | [Dbias](https://github.com/dreji18/Fairness-in-AI)       | Detect and mitigate biases in NLP tasks. The model is an end-to-end framework that takes data into raw form, preprocesses it, detects various types of biases, and mitigates them. The output is text free from bias. |
| Perspective API | [Perspective API by Google Jigsaw](https://www.perspectiveapi.com) | A tool created by Google Jigsaw that detects toxicity in text. It generates a probability of toxicity for a given text input and is widely used in research on mitigating toxic content in AI. |
| Aequitas        | [Aequitas Bias Audit Toolkit](https://dsapp.uchicago.edu/projects/aequitas) | An open-source toolkit designed to audit fairness and detect bias in machine learning models. Aequitas helps data scientists and policymakers understand and mitigate bias, including in large language models (LLMs). |
|LLMeBench|[LLMeBench](https://github.com/qcri/LLMeBench)|The framework currently supports evaluation of a variety of NLP tasks using three model providers: OpenAI (e.g., GPT), HuggingFace Inference API, and Petals (e.g., BLOOMZ); it can be seamlessly customized for any NLP task, LLM model and dataset, regardless of language.|


## General tools

| General Tool                         | Link                                                       | Author            | Description                                                                                                                                                                                                                                                                                                                |
|--------------------------------------|------------------------------------------------------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| responsible-ai-toolbox               | [responsible-ai-toolbox](https://github.com/microsoft/responsible-ai-toolbox) | Microsoft          | Responsible AI Toolbox is a suite of tools providing model and data exploration and assessment user interfaces and libraries that enable a better understanding of AI systems. These interfaces and libraries empower developers and stakeholders of AI systems to develop and monitor AI more responsibly and take better data-driven actions. |
| AIF360                               | [AIF360](https://github.com/Trusted-AI/AIF360)           | IBM               | A comprehensive set of fairness metrics for datasets and machine learning models, explanations for these metrics, and algorithms to mitigate bias in datasets and models.                                                                                                                                               |
| fairlearn                            | [fairlearn](https://github.com/fairlearn/fairlearn)       | fairlearn         | A Python package to assess and improve the fairness of machine learning models.                                                                                                                                                                                                                                          |
| The LinkedIn Fairness Toolkit (LiFT) | [LiFT](https://github.com/linkedin/LiFT)                 | LinkedIn          | A Scala/Spark library that enables the measurement of fairness in large scale machine learning workflows.                                                                                                                                                                                                                |
| Responsibly                          | [Responsibly](https://github.com/ResponsiblyAI/responsibly) | ResponsiblyAI     | Toolkit for auditing and mitigating bias and fairness of machine learning systems.                                                                                                                                                                                                                                        |
| ml-fairness-framework                | [ml-fairness-framework](https://github.com/firmai/ml-fairness-framework) | firmai            | FairPut - Machine Learning Fairness Framework with LightGBM — Explainability, Robustness, Fairness.                                                                                                                                                                                                                      |
| TruLens                              | [TruLens](https://github.com/truera/trulens)             | Truera            | Evaluation and tracking for LLM experiments.                                                                                                                                                                                                                                                                              |
| VeritasTool                          | [veritastool](https://github.com/mas-veritas2/veritastool) | MAS with partners  | Veritas Diagnosis Toolkit for Fairness Assessment; an interactive visualization tool.                                                                                                                                                                                                                                    |
## Explanation tools

| Tool                           | Description                                                                                                   | Link                                                                  |
|--------------------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| DALEX                          | moDel Agnostic Language for Exploration and eXplanation                                                      | [DALEX](https://github.com/ModelOriented/DALEX)                      |
| Alibi                          | An open source Python library aimed at machine learning model inspection and interpretation.                | [Alibi](https://github.com/SeldonIO/alibi)                           |
| AI Explainability 360          | An open-source library that supports interpretability and explainability of datasets and machine learning models. | [AI Explainability 360](https://github.com/Trusted-AI/AIF360)       |
| Shap                           | SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions   |[Shap](https://shap.readthedocs.io/en/latest/) |
| Lime                           | A project focused on explaining what machine learning classifiers (or models) are doing.                    | [Lime](https://github.com/marcotcr/lime)                            |
| InterpretML                   | An open-source package that incorporates state-of-the-art machine learning interpretability techniques under one roof. | [InterpretML](https://github.com/interpretml/interpret)            |
| Deep Visualization Toolbox     | Code required to run the Deep Visualization Toolbox and generate neuron-by-neuron visualizations using regularized optimization. | [Deep Visualization Toolbox](https://github.com/yosinski/DeepVis)   |
| Captum                         | A model interpretability and understanding library for PyTorch.                                            | [Captum](https://github.com/pytorch/captum)                          |
| Uncertainty Toolbox            | A toolbox for dealing with uncertainty in machine learning models.                                          | [Uncertainty Toolbox](https://github.com/Quentin-Desmoulin/uncertainty-toolbox) |
| Causal Inference 360          | A Python package for inferring causal effects from observational data.                                      | [Causal Inference 360](https://github.com/Trusted-AI/AIF360)        |


# Datasets
Many databases are proposed for fairness evaluation: 

- Probability-based Datasets:

These datasets are typically used to evaluate the probabilistic outputs of language models. They focus on how well a model can predict the likelihood of certain words or sequences of words occurring in a given context.
Such datasets often include a variety of text samples where the model's task is to assign probabilities to different possible continuations or completions. This helps in assessing the model's understanding of language patterns and its ability to generate coherent and contextually appropriate text.

- Generation-based Datasets:

These datasets are designed to evaluate the text generation capabilities of language models. They provide prompts or initial text inputs, and the model is tasked with generating extended text based on these inputs.
The focus is on the quality, coherence, and creativity of the generated text, as well as its adherence to any specified constraints or guidelines. These datasets help in assessing how well a model can produce human-like text and its potential biases in generation.
![image](https://github.com/user-attachments/assets/bbb8c35c-7ac0-4dec-9216-ff73c7232161) 
Figure from "Fairness in Large Language Models: A Taxonomic Survey"

Readers can find more information from https://github.com/i-gallegos/Fair-LLM-Benchmark

# Papers

## Survey papers
- Zuqiang Chu, Zichong Wang, Wenbin Zhang, **Fairness in Large Language Models: A Taxonomic Survey**, 2024, SIGKDD Explorations, 26(1), 34-48. 10.1145/3682112.3682117
- Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md. Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Ruiyi Zhang, Nesreen K. Ahmed, **Bias and Fairness in Large Language Models: A Survey**, 2023, 10.48550/arxiv.2309.00770
- Yingji Li, Mengnan Du, Rui Song, Xin Wang, Ying Wang, **A Survey on Fairness in Large Language Models**, 2024, arXiv, Preprint. Link
- Zhou Ren, Bias and Unfairness in Information Retrieval Systems: New Challenges in the LLM Era, 2023, arXiv, Preprint. Link
- Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi, Supryadi, Linhao Yu, Yan Liu, Jiaxuan Li, Bojian Xiong, Deyi Xiong, Large Language Models are not Fair Evaluators, 2024, arXiv, Preprint. Link
- A.G. Elrod, Uncovering Theological and Ethical Biases in LLMs, 2024, 10.7146/hn.v9i1.143407
- Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Rui Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, Hanguang Li, **Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models' Alignment**, 2023, 10.48550/arxiv.2308.05374
- Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ionut Stoica, Fairness in Serving Large Language Models, 2023, 10.48550/arxiv.2401.00588
- Thang Viet Doan, Zhibo Chu, Zichong Wang, Wenbin Zhang, Fairness Definitions in Language Models Explained, 2024, 10.48550/arxiv.2407.18454
- Shivani Kapania, Ruiyi Wang, Toby Jia-Jun Li, Tianshi Li, Hong Shen, "I'm categorizing LLM as a productivity tool": Examining ethics of LLM use in HCI research practices, 2024, 10.48550/arxiv.2403.19876


## General papers
- Abdelrahman Zayed, Gonçalo Mordido, Samira Shabanian, Ioana Baldini, Sarath Chandar, Fairness-Aware Structured Pruning in Transformers, 2024, 10.1609/aaai.v38i20.30256
- Adam X. Yang, Maxime Robeyns, Thomas Coste, Jun Wang, Haitham Bou Ammar, Laurence Aitchison, Bayesian Reward Models for LLM Alignment, 2024, 10.48550/arxiv.2402.13210
- Alaina N. Talboy, Elizabeth Fuller, Challenging the appearance of machine intelligence: Cognitive bias in LLMs, 2023, 10.48550/arXiv.2304.01358
- Amir Taubenfeld, Yaniv Dover, Roi Reichart, Ariel Goldstein, Systematic Biases in LLM Simulations of Debates, 2024, 10.48550/arxiv.2402.04049
- Aounon Kumar, Chirag Agarwal, Shilpa Kowdley Srinivas, Soheil Feizi, Hima Lakkaraju, Certifying LLM Safety against Adversarial Prompting, 2023, 10.48550/arxiv.2309.02705
- Bharat Prakash, Tim Oates, Tinoosh Mohsenin, LLM Augmented Hierarchical Agents, 2023, 10.48550/arxiv.2311.05596
- Boning Zhang, Chengxi Li, Kai Fan, MARIO Eval: Evaluate Your Math LLM with your Math LLM-A mathematical dataset evaluation toolkit, 2024, 10.48550/arxiv.2404.13925
- Canyu Chen, Kai Shu, Can LLM-Generated Misinformation Be Detected?, 2023, 10.48550/arxiv.2309.13788
- Chahat Raj, Anjishnu Mukherjee, Aylin Çalışkan, Antonios Anastasopoulos, Ziwei Zhu, Breaking Bias, Building Bridges: Evaluation and Mitigation of Social Biases in LLMs via Contact Hypothesis, 2024, 10.48550/arxiv.2407.02030
- Chen Xu, Wenjie Wang, Yuxin Li, Liang Pang, Jun Xu, Tat-Seng Chua, Do LLMs Implicitly Exhibit User Discrimination in Recommendation? An Empirical Study, 2023, 10.48550/arxiv.2311.07054
- Chris Bopp, Anne Foerst, Brian Kellogg, The Case for LLM Workshops, 2024, 10.1145/3626252.3630941
- Derek Snow, FairPut: A Light Framework for Machine Learning Fairness with LightGBM, 2020, 10.2139/SSRN.3619715
- Dong Huang, Qi Bu, Jie Zhang, Xiaofei Xie, Junjie Chen, Heming Cui, Bias Assessment and Mitigation in LLM-based Code Generation, 2023, 10.48550/arxiv.2309.14345
- Dylan Bouchard, An Actionable Framework for Assessing Bias and Fairness in Large Language Model Use Cases, 2024, 10.48550/arxiv.2407.10853
- Elinor Poole-Dayan, Deb Roy, Jad Kabbara, LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users, 2024, 10.48550/arxiv.2406.17737
- Erblin Isaku, Christoph Laaber, Hassan Sartaj, Shaukat Ali, Thomas Schwitalla, Jan F. Nygård, LLMs in the Heart of Differential Testing: A Case Study on a Medical Rule Engine, 2024, 10.48550/arxiv.2404.03664
- Fairness of ChatGPT, Fairness of ChatGPT, 2023, 10.48550/arxiv.2305.18569
- Garima Chhikara, Anurag Sharma, Kripabandhu Ghosh, Abhijnan Chakraborty, Few-Shot Fairness: Unveiling LLM's Potential for Fairness-Aware Classification, 2024, 10.48550/arxiv.2402.18502
- Grant Wilkins, Srinivasan Keshav, Richard Mortier, Offline Energy-Optimal LLM Serving: Workload-Based Energy Models for LLM Inference on Heterogeneous Systems, 2024, 10.48550/arxiv.2407.04014
- Guanqun Bi, Lei Shen, Yuqiang Xie, Yanan Cao, Tiangang Zhu, Xiaodong He, A Group Fairness Lens for Large Language Models, 2023, 10.48550/arxiv.2312.15478
- Haoxiang Wang, Yong Lin, Wei Xiong, Rui Yang, Shizhe Diao, Shuang Qiu, Han Zhao, Tong Zhang, Arithmetic Control of LLMs for Diverse User Preferences: Directional Preference Alignment with Multi-Objective Rewards, 2024, 10.48550/arxiv.2402.18571
- Henrique Da Silva Gameiro, LLM Detectors, 2024, 10.1007/978-3-031-54827-7_22
- Hongbin Sun, Yurong Chen, Siwei Wang, Wei Chen, Xiaotie Deng, Mechanism Design for LLM Fine-tuning with Multiple Reward Models, 2024, 10.48550/arxiv.2405.16276
- Ijsrem Journal, Developing A Fair Hiring Algorithm Using LLMs, 2023, 10.55041/ijsrem27101
- Iria Estévez-Ayres, Patricia Callejo, M.Á. Hombrados-Herrera, Carlos Alario-Hoyos, Carlos Delgado Kloos, Evaluation of LLM Tools for Feedback Generation in a Course on Concurrent Programming, 2024, 10.1007/s40593-024-00406-0
- Jizhi Zhang, Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation, 2023, 10.48550/arxiv.2305.07609
- Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He, Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation, 2023, 10.1145/3604915.3608860
- Jujia Zhao, Wenjie Wang, Chen Xu, Zhaochun Ren, See-kiong Ng, Tat-seng Chua, LLM-based Federated Recommendation, 2024, 10.48550/arxiv.2402.09959
- Kausik Lakkaraju, Sara E Jones, Sai Krishna Revanth Vuruma, Vishal Pallagani, Bharath Muppasani, Biplav Srivastava, LLMs for Financial Advisement: A Fairness and Efficacy Study in Personal Decision Making, 2023, 10.1145/3604237.3626867
- Jingling Li, Zeyu Tang, Xiaoyu Liu, Peter Spirtes, Kun Zhang, Liu Leqi, Yang Liu, Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework, 2024, 10.48550/arxiv.2403.08743
- Jessica Maria Echterhoff, Yao Liu, Abeer Alessa, Julian McAuley, Zexue He, Cognitive Bias in High-Stakes Decision-Making with LLMs, 2024, 10.48550/arxiv.2403.00811
- Jie Wang, LLM-Oracle Machines, 2024, 10.48550/arxiv.2406.12213
- Jin A. Shin, Henry Song, H.J. Lee, Soyeong Jeong, Jong C. Park, Ask LLMs Directly, "What shapes your bias?": Measuring Social Bias in Large Language Models, 2024, 10.48550/arxiv.2406.
- Jingling Li, Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework, 2024, 10.48550/arxiv.2403.08743
- Jingtong Gao, Chen Bin, Xiangyu Zhao, Weiwen Liu, Xiangyang Li, Yichao Wang, Zijian Zhang, Wanyu Wang, Yuyang Ye, Shawn D. Lin, Huifeng Guo, Ruiming Tang, LLM-enhanced Reranking in Recommender Systems, 2024, 10.48550/arxiv.2406.12433

- Keito Inoshita, Assessment of Conflict Structure Recognition and Bias Impact in Japanese LLMs, 2024, 10.1109/times-icon61890.2024.10630720
- Kun Zhou, Yutao Zhu, Zhipeng Chen, Wentong Chen, Wayne Xin Zhao, Xu Chen, Yankai Lin, Jinhui Wen, Jiawei Han, Don't Make Your LLM an Evaluation Benchmark Cheater, 2023, 10.48550/arxiv.2311.01964
- Kuniaki Saito, Kihyuk Sohn, Chen-Yu Lee, Yoshitaka Ushiku, Unsupervised LLM Adaptation for Question Answering, 2024, 10.48550/arxiv.2402.12170
- Lauren Rhue, Sofie Goethals, Arun Sundararajan, Evaluating LLMs for Gender Disparities in Notable Persons, 2024, 10.48550/arxiv.2403.09148
- Lieh-Chiu Lin, H. Alex Brown, Kenji Kawaguchi, Michael Shieh, Single Character Perturbations Break LLM Alignment, 2024, 10.48550/arxiv.2407.03232
- Loïc Maréchal, Daniel Celeny, Insurance Outlook for LLM-Induced Risk, 2024, 10.1007/978-3-031-54827-7_15
- Luc Patiny, Guillaume Godin, Automatic extraction of FAIR data from publications using LLM, 2023, 10.26434/chemrxiv-2023-05v1b
- Lucas Potter, Xavier–Lewis Palmer, Post-LLM Academic Writing Considerations, 2023, 10.1007/978-3-031-47448-4_12
- Luyang Lin, Lingzhi Wang, Jinsong Guo, Kam-Fai Wong, Investigating Bias in LLM-Based Bias Detection: Disparities between LLMs and Human Perception, 2024, 10.48550/arxiv.2403.14896
- M. Kamruzzaman, M. M. I. Shovon, Gene Louis Kim, Investigating Subtler Biases in LLMs: Ageism, Beauty, Institutional, and Nationality Bias in Generative Models, 2023, 10.48550/arxiv.2309.08902
- Marjan Fariborz, Mahyar Samani, Pouya Fotouhi, Roberto Proietti, Il-Min Yi, Venkatesh Akella, Jason Lowe-Power, Samuel Palermo, S. J. B. Yoo, LLM: Realizing Low-Latency Memory by Exploiting Embedded Silicon Photonics for Irregular Workloads, 2022, 10.1007/978-3-031-07312-0_3
- Mark Pock, Andre Ye, Jared Moore, LLMs grasp morality in concept, 2023, 10.48550/arxiv.2311.02294
- Mohsen Fayyaz, Fan Yin, Jiao Sun, Nanyun Peng, Evaluating Human Alignment and Model Faithfulness of LLM Rationale, 2024, 10.48550/arxiv.2407.00219
- Nat McAleese, Rai Michael Pokorny, Juan Felipe Ceron Uribe, Evgenia Nitishinskaya, Maja Trębacz, Jan Leike, LLM Critics Help Catch LLM Bugs, 2024, 10.48550/arxiv.2407.00215
- Praise Chinedu-Eneh, Trung T. Nguyen, Elephant: LLM System for Accurate Recantations, 2024, 10.1109/ccwc60891.2024.10427759
- Qizhang Feng, Siva Rajesh Kasa, Hyokun Yun, Choon Hui Teo, Sravan Bodapati, Exposing Privacy Gaps: Membership Inference Attack on Preference Data for LLM Alignment, 2024, 10.48550/arxiv.2407.06443
- Raphael Poulain, Hamed Fayyaz, Rahmatollah Beheshti, Bias patterns in the application of LLMs for clinical decision support: A comprehensive study, 2024, 10.48550/arxiv.2404.15149
- Rifki Afina Putri, Faiz Ghifari Haznitrama, Dea Adhista, Alice Oh, Can LLM Generate Culturally Relevant Commonsense QA Data? Case Study in Indonesian and Sundanese, 2024, 10.48550/arxiv.2402.17302
- Rohan Alexander, Lindsay Katz, Caitlin Moore, Zvi Schwartz, Evaluating the Decency and Consistency of Data Validation Tests Generated by LLMs, 2023, 10.48550/arxiv.2310.01402
- Roman Koshkin, K. Sudoh, Satoshi Nakamura, TransLLaMa: LLM-based Simultaneous Translation System, 2024, 10.48550/arxiv.2402.04636
- Rongsheng Wang, Hao Chen, Ruizhe Zhou, Han Ma, Yaofei Duan, Yanlan Kang, Songhua Yang, Baoyu Fan, Tao Tan, LLM-Detector: Improving AI-Generated Chinese Text Detection with Open-Source LLM Instruction Tuning, 2024, 10.48550/arxiv.2402.01158
- Satya Vart Dwivedi, Sanjukta Ghosh, Shivam Dwivedi, Breaking the Bias: Gender Fairness in LLMs Using Prompt Engineering and In-Context Learning, 2023, 10.21659/rupkatha.v15n4.10
- Shaina Raza, Shardul Ghuge, Chen Ding, Elham Dolatabadi, Deval Pandya, FAIR Enough: Develop and Assess a FAIR-Compliant Dataset for Large Language Model Training?, 2024, 10.1162/dint_a_00255
- Shashank Gupta, Vaishnavi Shrivastava, Ameet Shridhar Deshpande, Ashwin Kalyan, Peter Clark, Ashish Sabharwal, Tushar Khot, Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs, 2023, 10.48550/arxiv.2311.04892
- Shubh Goyal, Medha Hira, Shubham Mishra, Sukriti Goyal, Arnav Goel, Niharika Dadu, DB Kirushikesh, Sameep Mehta, Nishtha Madaan, LLMGuard: Guarding against Unsafe LLM Behavior, 2024, 10.1609/aaai.v38i21.30566
- Song Wang, Peng Wang, Tong Zhou, Yiqi Dong, Zhen Tan, Jundong Li, CEB: Compositional Evaluation Benchmark for Fairness in Large Language Models, 2024, 10.48550/arxiv.2407.02408
- Steven J. DeRose, Can LLMs help with XML?, 2024, 10.4242/balisagevol29.derose01
- Subhabrata Majumdar, Standards for LLM Security, 2024, 10.1007/978-3-031-54827-7_25
- Sumuk Shashidhar, Abhinav Chinta, Vaibhav Sahai, Zhenhailong Wang, Heng Ji, Democratizing LLMs: An Exploration of Cost-Performance Trade-offs in Self-Refined Open-Source Models, 2023, 10.48550/arxiv.2310.07611
- Swanand Kadhe, Anisa Halimi, Ambrish Rawat, Nathalie Baracaldo, FairSISA: Ensemble Post-Processing to Improve Fairness of Unlearning in LLMs, 2023, 10.48550/arxiv.2312.07420
- Virginia K. Felkner, John A. Thompson, Jonathan May, GPT is Not an Annotator: The Necessity of Human Annotation in Fairness Benchmark Construction, 2024, 10.48550/arxiv.2405.15760
- Vyas Raina, Adian Liusie, Mark Gales, Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment, 2024, 10.48550/arxiv.2402.14016
- Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, Yongbin Li, Wider and Deeper LLM Networks are Fairer LLM Evaluators, 2023, 10.48550/arxiv.2308.01862
- Xinru Wang, Sajjadur Rahman, Kushan Mitra, Zhengjie Miao, Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels, 2024, 10.1145/3613904.3641960
- Yashar Deldjoo, Tommaso di Noia, CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System, 2024, arXiv, Preprint. Link
- Yashar Deldjoo, Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency, 2024, arXiv, Preprint. Link
- Yifan Zeng, Ojas Tendolkar, Raymond Baartmans, Qingyun Wu, Huazheng Wang, Lizhong Chen, LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking, 2024, 10.48550/arxiv.2406.00231
- Yijiang River Dong, Tiancheng Hu, Nigel Collier, Can LLM be a Personalized Judge?, 2024, 10.48550/arxiv.2406.11657
- Yixin Wan, George Pu, Jiao Sun, Aparna Garimella, Kai-Wei Chang, Nanyun Peng, "Kelly is a Warm Person, Joseph is a Role Model": Gender Biases in LLM-Generated Reference Letters, 2023, 10.48550/arxiv.2310.09219
- Yuan Wang, Xuyang Wu, Hsin-Tai Wu, Zhiqiang Tao, Yi Fang, Do Large Language Models Rank Fairly? An Empirical Study on the Fairness of LLMs as Rankers, 2024, 10.48550/arxiv.2404.03192
- Yunqi Li, Yongfeng Zhang, Fairness of ChatGPT, 2023, 10.48550/arXiv.2305.18569
- Zelong Li, Wen Hua, Hao Wang, He Zhu, Yongfeng Zhang, Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents, 2024, 10.48550/arxiv.2402.00798
- Zhengyu Hu, Linxin Song, Jieyu Zhang, Zheyuan Xiao, Jingang Wang, Frank Jm Geurts, Jieyu Zhao, Jingbo Zhou, Rethinking LLM-based Preference Evaluation, 2024, 10.48550/arxiv.2407.01085

# Other Useful Links


| **Tool Name**                               | **Link**                                                     | **Functionality**                                                                                                                                                              |
|---------------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Awesome Machine Learning Interpretability** | [GitHub Link](https://github.com/jphall663/awesome-machine-learning-interpretability) | A curated list of resources, libraries, and tools focused on interpreting machine learning models. It likely includes frameworks for model explanation, visualization techniques, and best practices for enhancing model transparency. |
| **Awesome Trustworthy Deep Learning**       | [GitHub Link](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning) | A collection of resources aimed at ensuring the trustworthiness of deep learning models. This may cover topics like model robustness, reliability, fairness, and security in deep learning applications.                      |
| **Fair LLM Benchmark**                      | [GitHub Link](https://github.com/i-gallegos/Fair-LLM-Benchmark) | A benchmark suite designed to evaluate fairness in Large Language Models (LLMs). It likely includes standardized tests and metrics to assess and compare the fairness of different LLMs across various tasks and datasets.       |
| **LLM-IR Bias Fairness Survey**              | [GitHub Link](https://github.com/KID-22/LLM-IR-Bias-Fairness-Survey) | A survey repository that reviews and summarizes research on bias and fairness in Large Language Models and Information Retrieval systems. It may include literature reviews, analysis of existing studies, and identified gaps in the field. |
| **Awesome Fairness Papers**                  | [GitHub Link](https://github.com/uclanlp/awesome-fairness-papers) | A comprehensive list of influential research papers on fairness in machine learning and AI. It serves as a resource for academics and practitioners looking to understand the latest advancements and methodologies in fair AI.       |
| **Awesome Fairness in AI**                   | [GitHub Link](https://github.com/datamllab/awesome-fairness-in-ai) | A curated repository of tools, libraries, datasets, and research focused on fairness in artificial intelligence. It likely includes resources for implementing fair algorithms, evaluating bias, and promoting equitable AI practices. |
| **Awesome ML Fairness**                      | [GitHub Link](https://github.com/brandeis-machine-learning/awesome-ml-fairness) | A collection of resources dedicated to fairness in machine learning. This may encompass algorithms for bias mitigation, fairness metrics, tutorials, and frameworks to help developers build equitable ML models.                      |
| **Awesome Responsible AI**                   | [GitHub Link](https://github.com/AthenaCore/AwesomeResponsibleAI) | A curated list of resources, tools, and frameworks aimed at promoting responsible AI development and deployment. It likely covers areas such as ethical AI, transparency, accountability, and best practices for sustainable AI systems. |


