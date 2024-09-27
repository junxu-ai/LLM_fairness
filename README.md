# Fairness in LLM
Collection of papers, tools, datasets for fairness of LLM

# Introduction
Fairness in Large Language Models (LLMs) is a critical area of research, as these models are increasingly used in applications that impact diverse populations. The challenge lies in ensuring that LLMs do not perpetuate or exacerbate biases, particularly against marginalized groups. This involves understanding the sources of bias, developing metrics for assessing fairness, and implementing strategies to mitigate unfair outcomes. The following sections explore these aspects in detail, drawing insights from recent research.

## Sources of Bias in LLMs
- Data and Training Bias: LLMs are trained on vast datasets that may contain historical biases, leading to biased model outputs. These biases can manifest in various forms, such as gender, race, and age discrimination.
- Model Architecture and Design: The design of LLMs can inherently favor certain types of data or interactions, which may not be representative of all user groups.
- Use Case Specific Bias: Different applications of LLMs, such as text classification or information retrieval, can introduce unique biases based on how the models are used and the context in which they operate.

## Assessing Fairness in LLMs
- Fairness Metrics: Various metrics have been proposed to evaluate fairness in LLMs, including counterfactual metrics and stereotype classifiers. These metrics help in quantifying bias and assessing the fairness of model outputs.
- Group Fairness Evaluation: Evaluating LLMs through a group fairness lens involves analyzing how different social groups are represented and treated by the models. This approach uses datasets like GFair to assess biases across multiple dimensions.
- Empirical Studies: Studies using datasets such as the TREC Fair Ranking dataset provide empirical benchmarks for evaluating fairness in LLMs, particularly in ranking tasks.

## Mitigation Strategies
- Prompt Engineering and In-Context Learning: Techniques such as prompt engineering and in-context learning have been shown to reduce biases in LLM outputs by guiding the models to generate more equitable content.
- Remediation Techniques: Various strategies, including data resampling, model regularization, and post-processing techniques, have been proposed to mitigate biases in LLM-based applications.
- Fairness-Aware Frameworks: Frameworks that incorporate fairness regulations and definitions into the model training and evaluation process can help ensure more inclusive and fair outcomes.

## Challenges and Future Directions
- Complexity of Fairness in LLMs: The inherent complexity of LLMs and their diverse applications make it challenging to apply traditional fairness frameworks. This necessitates the development of new guidelines and iterative processes involving stakeholders to achieve fairness in specific use cases.
- Scalable Solutions: As LLMs continue to evolve, scalable solutions that leverage AI's general-purpose capabilities to address fairness challenges are needed.
- Fairness in Serving LLMs: Ensuring fairness in LLM inference services involves addressing challenges related to request scheduling and resource utilization, with novel algorithms like the Virtual Token Counter (VTC) showing promise in this area.

While significant progress has been made in understanding and addressing fairness in LLMs, challenges remain. The complexity of human-AI interactions and the diverse contexts in which LLMs are deployed require ongoing research and innovation. Moreover, achieving fairness is not just a technical challenge but also a social one, necessitating collaboration among developers, users, and stakeholders to create AI systems that are equitable and just.


# Papers

## Survey papers
- Zuqiang Chu, Zichong Wang, Wenbin Zhang, Fairness in Large Language Models: A Taxonomic Survey, 2024, SIGKDD Explorations, 26(1), 34-48. 10.1145/3682112.3682117
- Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md. Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Ruiyi Zhang, Nesreen K. Ahmed, Bias and Fairness in Large Language Models: A Survey, 2024, arXiv, Preprint. Link
- Zhou Ren, Bias and Unfairness in Information Retrieval Systems: New Challenges in the LLM Era, 2023, arXiv, Preprint. Link
- Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi, Supryadi, Linhao Yu, Yan Liu, Jiaxuan Li, Bojian Xiong, Deyi Xiong, Large Language Models are not Fair Evaluators, 2024, arXiv, Preprint. Link
- A.G. Elrod, Uncovering Theological and Ethical Biases in LLMs, 2024, 10.7146/hn.v9i1.143407
- Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Rui Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, Hanguang Li, Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models' Alignment, 2023, 10.48550/arxiv.2308.05374
- Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ionut Stoica, Fairness in Serving Large Language Models, 2023, 10.48550/arxiv.2401.00588
- Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md. Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Ruiyi Zhang, Nesreen K. Ahmed, Bias and Fairness in Large Language Models: A Survey, 2023, 10.48550/arxiv.2309.00770
- Thang Viet Doan, Zhibo Chu, Zichong Wang, Wenbin Zhang, Fairness Definitions in Language Models Explained, 2024, 10.48550/arxiv.2407.18454
- Shivani Kapania, Ruiyi Wang, Toby Jia-Jun Li, Tianshi Li, Hong Shen, "I'm categorizing LLM as a productivity tool": Examining ethics of LLM use in HCI research practices, 2024, 10.48550/arxiv.2403.19876
- Yingji Li, Mengnan Du, Rui Song, Xin Wang, Ying Wang, A Survey on Fairness in Large Language Models, 2024, arXiv, Preprint. Link

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


