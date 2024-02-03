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
10 Jul 2023|Large Language Models as General Pattern Machines](https://arxiv.org/abs/2307.04721)|Stanford University, Google DeepMind, etc.|CoRL'23|General|Forecasting, etc.|GPT-3, PaLM
29 Aug 2023|[Where Would I Go Next? Large Language Models as Human Mobility Predictors](https://arxiv.org/abs/2308.15197)|University College London, University of Liverpool|Preprint|Mobility|Forecasting|GPT-3.5
7 Oct 2023|[Large Language Models for Spatial Trajectory Patterns Mining](https://arxiv.org/abs/2310.04942)|Emory University|Preprint|Mobility|Anomaly Detection|GPT-3.5, GPT-4, Claude-2
11 Oct 2023|[Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820)|NYU, CMU|NeurIPS'23|General|Forecasting|GPT-3, LLaMA-2 
26 Oct 2023|[Utilizing Language Models for Energy Load Forecasting](https://arxiv.org/abs/2310.17788)|University of New South Wales|BuildSys'23|Energy|Forecasting|Bart, Bigbird, Pegasus

### Citation

If you find this useful, please cite our paper: "Large Language Models for Time Series: A Survey"
