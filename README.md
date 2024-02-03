# awesome-llm-time-series

Keep updating the repo; stay tuned!

![](./taxonomy.png)
Left: Taxonomy of LLMs for time series analysis. If we outline typical LLM-driven NLP pipelines in five stages - input text, tokenization, embedding, LLM, output - then each category of our taxonomy targets one specific stage in this pipeline:
* Prompting (input stage) treats time series data as raw text and directly prompts LLMs with time series;
* Time Series Quantization (tokenization stage) discretizes time series as special tokens for LLMs to process;
* Alignment (embedding stage) designs time series encoder to align time series embeddings with language space;
* Vision as Bridge (LLM stage) connects time series with Vision-Lanuage Models (VLM) by employing visual representations as a bridge;
* Tool Integration (output stage) adopts language models to output tools to benefit time series analysis.

Right: Representative works for each category, sorted by their publication dates. The use of arrows indicates that later works build upon earlier studies. Dark(light)-colored boxes represent billion(million)-parameter models. Icons to the left of the text boxes represent the application domains of domain-specific models.

- [Taxonomy](#taxonomy)
  - [Prompting](#prompting)
  - [Quantization](#quantization)
  - [Alignment](#alignment)
  - [Vision](#vision)
  - [Tool](#tool)
- [Datasets](#datasets)
    
## Taxonomy

### Prompting

Date|Paper|Institute|Conference|Domain|Task|LLM
----|---------------------|----|----|----|----|----
11 Sep 2022|[Leveraging Language Foundation Models for Human Mobility Forecasting](https://arxiv.org/abs/2209.05479)|University of New South Wales|SIGSPATIAL'22|Mobility|Forecasting|Bert, BoBERTa, GPT-2, etc.
20 Sep 2022|[PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting](https://arxiv.org/abs/2210.08964)|University of New South Wales|TKDE|General|Forecasting|Bart, BigBird, RoBERTa, etc.
19 Oct 2022|[TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/abs/2210.10723)|MIT, University of Münster|AISTATS'23|Table|Classification|T0, GPT-3
30 Mar 2023|[BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)|Bloomberg, Johns Hopkins University|Preprint|Finance|Sentiment Analysis, NER, QA, etc.|BloombergGPT
10 Apr 2023|[The Wall Street Neophyte: A Zero-Shot Analysis of ChatGPT Over MultiModal Stock Movement Prediction Challenges](https://arxiv.org/abs/2304.05351)|Wuhan University, Southwest Jiaotong University, etc.|Preprint|Finance|Forecasting|ChatGPT
24 May 2023|[Large Language Models are Few-Shot Health Learners](https://arxiv.org/abs/2305.15525)|Google|Preprint|Health|Classification, Regression|PaLM
10 Jul 2023|[Large Language Models as General Pattern Machines](https://arxiv.org/abs/2307.04721)|Stanford University, Google DeepMind, etc.|CoRL'23|General|Forecasting, Translation, etc.|GPT-3, PaLM
29 Aug 2023|[Where Would I Go Next? Large Language Models as Human Mobility Predictors](https://arxiv.org/abs/2308.15197)|University College London, University of Liverpool|Preprint|Mobility|Forecasting|GPT-3.5
7 Oct 2023|[Large Language Models for Spatial Trajectory Patterns Mining](https://arxiv.org/abs/2310.04942)|Emory University|Preprint|Mobility|Anomaly Detection|GPT-3.5, GPT-4, Claude-2
11 Oct 2023|[Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820)|NYU, CMU|NeurIPS'23|General|Forecasting|GPT-3, LLaMA-2 
26 Oct 2023|[Utilizing Language Models for Energy Load Forecasting](https://arxiv.org/abs/2310.17788)|University of New South Wales|BuildSys'23|Energy|Forecasting|Bart, Bigbird, Pegasus

### Quantization

![](./quantization.png)

Left: VQ-VAE based quantization; Right: K-Means based quantization

Date|Paper|Institute|Conference|Domain|Task|LLM
----|---------------------|----|----|----|----|----
7 Sep 2022|[AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143)|Google|Preprint|Audio|Generation|w2v-BERT
30 Sep 2022|[AudioGen: Textually Guided Audio Generation](https://arxiv.org/abs/2209.15352)|Meta, The Hebrew University of Jerusalem|ICLR'23|Audio|Generation|Transformer
9 Mar 2023|[Text-to-ECG: 12-Lead Electrocardiogram Synthesis conditioned on Clinical Text Reports](https://arxiv.org/abs/2303.09395)|KAIST, Medical AI Inc., etc.|ICASSP'23|Health|Generation|Transformer
18 May 2023|[SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities](https://aclanthology.org/2023.findings-emnlp.1055/)|Fudan University|EMNLP'23 Findings|Audio|Generation, Translation|LLaMA
25 May 2023|[VioLA: Unified Codec Language Models for Speech Recognition, Synthesis, and Translation](https://arxiv.org/abs/2305.16107)|Microsoft|Preprint|Audio|Generation, Translation|Transformer
19 Jun 2023|[Temporal Data Meets LLM -- Explainable Financial Time Series Forecasting](https://arxiv.org/abs/2306.11025)|Amazon|Preprint|Finance|Forecasting|GPT-4, Open LLaMA
22 Jun 2023|[AudioPaLM: A Large Language Model That Can Speak and Listen](https://arxiv.org/abs/2306.12925)|Google|Preprint|Audio|Generation, Translation|PaLM-2
15 Sept 2023|[Modeling Time Series as Text Sequence A Frequency-vectorization Transformer for Time Series Forecasting](https://openreview.net/forum?id=N1cjy5iznY)|Anonymous|OpenReview|General|Forecasting|Transformer
22 Sept 2023|[Time Series Modeling at Scale: A Universal Representation Across Tasks and Domains](https://openreview.net/forum?id=SZErAetdMu)|Anonymous|OpenReview|General|Forecasting, Classification, etc.|Transformer
25 Sep 2023|[DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation](https://arxiv.org/abs/2309.14030)|University of Technology Sydney, The University of Sydney|NeurIPS'23|Health|Text Generation|BART
1 Oct 2023|[UniAudio: An Audio Foundation Model Toward Universal Audio Generation](https://arxiv.org/abs/2310.00704)|The Chinese University of Hong Kong, CMU, etc.|Preprint|Audio|Generation|Transformer


### Citation

If you find this useful, please cite our paper: "Large Language Models for Time Series: A Survey"
