# awesome-llm-time-series

Tracking papers on "LLM for time series analysis", more details in [Large Language Models for Time Series: A Survey](https://arxiv.org/abs/2402.01801).

[New🔥] Our survey paper is accepted to IJCAI 2024, survey track!

![](./taxonomy.png)
Left: Taxonomy of LLMs for time series analysis. If we outline typical LLM-driven NLP pipelines in five stages - input text, tokenization, embedding, LLM, output - then each category of our taxonomy targets one specific stage in this pipeline:
* Prompting (input stage) treats time series data as raw text and directly prompts LLMs with time series;
* Time Series Quantization (tokenization stage) discretizes time series as special tokens for LLMs to process;
* Aligning (embedding stage) designs time series encoder to align time series embeddings with language space;
* Vision as Bridge (LLM stage) connects time series with Vision-Lanuage Models (VLM) by employing visual representations as a bridge;
* Tool Integration (output stage) adopts language models to output tools to benefit time series analysis.

Right: Representative works for each category, sorted by their publication dates. The use of arrows indicates that later works build upon earlier studies. Dark(light)-colored boxes represent billion(million)-parameter models. Icons to the left of the text boxes represent the application domains of domain-specific models.

- [Taxonomy](#taxonomy)
  - [Prompting](#prompting)
  - [Quantization](#quantization)
  - [Aligning](#aligning)
  - [Vision](#vision)
  - [Tool](#tool)
- [Datasets](#datasets)
- [Citation](#citation)
    
## Taxonomy

### Prompting

Date|Paper|Institute|Conference|Domain|Task|LLM
----|---------------------|----|----|----|----|----
11 Sep 2022|[Leveraging Language Foundation Models for Human Mobility Forecasting](https://arxiv.org/abs/2209.05479)|University of New South Wales|SIGSPATIAL'22|Mobility|Forecasting|BERT, BoBERTa, GPT-2, etc.
20 Sep 2022|[PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting](https://arxiv.org/abs/2210.08964)|University of New South Wales|TKDE'23|General|Forecasting|BART, BigBird, RoBERTa, etc.
19 Oct 2022|[TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/abs/2210.10723)|MIT, University of Münster|AISTATS'23|Table|Classification|T0, GPT-3
30 Mar 2023|[BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)|Bloomberg, Johns Hopkins University|Preprint|Finance|Sentiment Analysis, NER, QA, etc.|BloombergGPT
10 Apr 2023|[The Wall Street Neophyte: A Zero-Shot Analysis of ChatGPT Over MultiModal Stock Movement Prediction Challenges](https://arxiv.org/abs/2304.05351)|Wuhan University, Southwest Jiaotong University, etc.|Preprint|Finance|Forecasting|ChatGPT
24 May 2023|[Large Language Models are Few-Shot Health Learners](https://arxiv.org/abs/2305.15525)|Google|Preprint|Health|Classification, Regression|PaLM
10 Jul 2023|[Large Language Models as General Pattern Machines](https://arxiv.org/abs/2307.04721)|Stanford University, Google DeepMind, etc.|CoRL'23|General|Forecasting, Translation, etc.|GPT-3, PaLM
29 Aug 2023|[Where Would I Go Next? Large Language Models as Human Mobility Predictors](https://arxiv.org/abs/2308.15197)|University College London, University of Liverpool|Preprint|Mobility|Forecasting|GPT-3.5
7 Oct 2023|[Large Language Models for Spatial Trajectory Patterns Mining](https://arxiv.org/abs/2310.04942)|Emory University|Preprint|Mobility|Anomaly Detection|GPT-3.5, GPT-4, Claude-2
11 Oct 2023|[Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820)|NYU, CMU|NeurIPS'23|General|Forecasting|GPT-3, LLaMA-2 
26 Oct 2023|[Utilizing Language Models for Energy Load Forecasting](https://arxiv.org/abs/2310.17788)|University of New South Wales|BuildSys'23|Energy|Forecasting|BART, Bigbird, Pegasus
16 Feb 2024|[Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities](https://arxiv.org/abs/2402.10835)| Rutgers University, Shanghai Jiao Tong University, etc.|Preprint|General|Forecasting|GPT-3.5, GPT-4, LLaMA-2
25 Feb 2024|[LSTPrompt: Large Language Models as Zero-Shot Time Series Forecasters by Long-Short-Term Prompting](https://arxiv.org/abs/2402.16132)|Georgia Institute of Technology, Microsoft Research Asia|Preprint|General|Forecasting|GPT-3.5, GPT-4
2 Mar 2024|[Evaluating Large Language Models as Virtual Annotators for Time-series Physical Sensing Data](https://arxiv.org/abs/2403.01133)|Indian Institute of Technology, Kharagpur, Nokia Bell Labs|Preprint|IoT|Classification|GPT-4

### Quantization

![](./quantization.png)

Left: VQ-VAE based quantization

Right: K-Means based quantization

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
25 Sep 2023|[DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation](https://arxiv.org/abs/2309.14030)|University of Technology Sydney, The University of Sydney|NeurIPS'23|Health|Generation|BART
1 Oct 2023|[UniAudio: An Audio Foundation Model Toward Universal Audio Generation](https://arxiv.org/abs/2310.00704)|The Chinese University of Hong Kong, CMU, etc.|Preprint|Audio|Generation|Transformer
12 Mar 2024|[Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)|Amazon Web Services, UC San Diego, etc.|Preprint|General|Forecasting|T5
19 Mar 2024|[Advancing Time Series Classification with Multimodal Language Modeling](https://arxiv.org/abs/2403.12371)|University of Science and Technology of China|Preprint|General|Classification|GPT-2
19 Mar 2024|[Learning Transferable Time Series Classifier with Cross-Domain Pre-training from Language Model](https://arxiv.org/abs/2403.12372)|University of Science and Technology of China, Kuaishou Technology|Preprint|General|Classification|BERT, GPT-2

### Aligning

![](./alignment.png)

Left: Aligning by similarity matching

Right: Aligning with large language models as backbones, where the output could be time series (e.g., forecasting) or text (e.g., EEG-to-text) depending on the downstream tasks.

Date|Paper|Institute|Conference|Domain|Task|LLM
----|---------------------|----|----|----|----|----
5 Dec 2021|[Open Vocabulary Electroencephalography-To-Text Decoding and Zero-shot Sentiment Classification](https://arxiv.org/abs/2112.02690)|UIUC|AAAI'22|Health|Translation, Classification|BART
29 Mar 2022|[WAVPROMPT: Towards Few-Shot Spoken Language Understanding with Frozen Language Models](https://arxiv.org/abs/2203.15863)|UIUC, MIT-IBM Watson AI Lab, etc.|Preprint|Audio|Classification|GPT-2
10 Aug 2022|[Can Brain Signals Reveal Inner Alignment with Human Languages?](https://arxiv.org/abs/2208.06348)|CMU, MIT, etc.|EMNLP'23 Findings|Health|Sentiment Analysis, Relation Detection|BERT
21 Jan 2023|[Transfer Knowledge from Natural Language to Electrocardiography: Can We Detect Cardiovascular Disease Through Language Models?](https://arxiv.org/abs/2301.09017)|CMU, Allegheny General Hospital, etc.|EACL'23 Findings|Health|Generation, Classification|BERT, BART
23 Feb 2023|[One Fits All:Power General Time Series Analysis by Pretrained LM](https://arxiv.org/abs/2302.11939)|Alibaba|NeurIPS'23|General|Forecasting, Classification, etc.|GPT-2
18 May 2023|[Listen, Think, and Understand](https://arxiv.org/abs/2305.10790)|MIT, MIT-IBM Watson AI Lab|ICLR'24|Audio|Classification, Captioning|Vicuna
16 Aug 2023|[TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series](https://arxiv.org/pdf/2308.08241.pdf)|Peking University, Alibaba Group|ICLR'24|General|Forecasting, Classification|BERT, GPT-2, ChatGLM, etc.
16 Aug 2023|[LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters](https://arxiv.org/abs/2308.08469)|National Yang Ming Chiao Tung University|Preprint|General|Forecasting|GPT-2
22 Aug 2023|[Music Understanding LLaMA: Advancing Text-to-Music Generation with Question Answering and Captioning](https://arxiv.org/abs/2308.11276)|Tencent, National University of Singapore|Preprint|Audio|QA, Captioning|LLaMA
6 Sep 2023|[ETP: Learning Transferable ECG Representations via ECG-Text Pre-training](https://arxiv.org/abs/2309.07145)|Imperial College London, The Ohio State University|Preprint|Health|Classification|BioClinicalBERT
19 Sep 2023|[End-to-End Speech Recognition Contextualization with Large Language Models](https://arxiv.org/abs/2309.10917)|Meta|Preprint|Audio|ASR|LLaMA
3 Oct 2023|[Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)|Monash University, Ant Group, etc.|ICLR'24|General|Forecasting|LLaMA
8 Oct 2023|[TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting](https://arxiv.org/abs/2310.04948)|University of Southern California, Google|ICLR'24|General|Forecasting|GPT-2
12 Oct 2023|[Lag-Llama: Towards Foundation Models for Time Series Forecasting](https://arxiv.org/abs/2310.08278)|Morgan Stanley, ServiceNow, etc.|NeurIPS'23 Workshop|General|Forecasting|LLaMA
15 Oct 2023|[UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting](https://arxiv.org/abs/2310.09751)|National University of Singapore, The Hong Kong University of Science and Technology|WWW'24|General|Forecasting|GPT-2
20 Oct 2023|[SALMONN: Towards Generic Hearing Abilities for Large Language Models](https://arxiv.org/abs/2310.13289)|Tsinghua University, ByteDance|ICLR'24|Audio|ASR, QA, etc.|Vicuna
27 Oct 2023|[JoLT: Jointly Learned Representations of Language and Time-Series](https://openreview.net/forum?id=UVF1AMBj9u&referrer=%5Bthe%20profile%20of%20Yifu%20Cai%5D(%2Fprofile%3Fid%3D~Yifu_Cai1))|CMU|NeurIPS'23 Workshop|Health|Captioning, QA|GPT-2, OPT
14 Nov 2023|[TENT: Connect Language Models with IoT Sensors for Zero-Shot Activity Recognition](https://arxiv.org/abs/2311.08245)|Nanyang Technological University|Preprint|IoT|Classification|CLIP
24 Nov 2023|[GATGPT: A Pre-trained Large Language Model with Graph Attention Network for Spatiotemporal Imputation](https://arxiv.org/abs/2311.14332)|University of Technology Sydney|Preprint|General|Imputation|GPT-2
11 Dec 2023|[Multimodal Pretraining of Medical Time Series and Notes](https://arxiv.org/abs/2312.06855)|Texas A&M University|ML4H'23|Health|Classification|BERT
18 Jan 2024|[Spatial-Temporal Large Language Model for Traffic Prediction](https://arxiv.org/abs/2401.10134)|Nanyang Technological University, Peking University, etc.|Preprint|Traffic|Forecasting|GPT-2, LLaMA-2
25 Jan 2024|[How Can Large Language Models Understand Spatial-Temporal Data?](https://arxiv.org/abs/2401.14192)|Dalian University of Technology|Preprint|General|Forecasting|GPT-2
26 Jan 2024|[Large Language Model Guided Knowledge Distillation for Time Series Anomaly Detection](https://arxiv.org/abs/2401.15123)|Zhejiang University|Preprint|General|Anomaly Detection|GPT-2
4 Feb 2024|[AutoTimes: Autoregressive Time Series Forecasters via Large Language Models](https://arxiv.org/abs/2402.02370)|Tsinghua University|Preprint|General|Forecasting|LLaMA, GPT-2, OPT
7 Feb 2024|[Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning](https://arxiv.org/abs/2402.04852)|The Chinese University of Hong Kong, Tongji University, etc.|Preprint|General|Forecasting, Anomaly Detection|GPT-2
9 Mar 2024|[S^2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting](https://arxiv.org/abs/2403.05798)|University of Connecticut, Morgan Stanley|Preprint|General|Forecasting|GPT-2
12 Mar 2024|[Taming Pre-trained LLMs for Generalised Time Series Forecasting via Cross-modal Knowledge Distillation](https://arxiv.org/abs/2403.07300)|Tsinghua University, Shenzhen University|Preprint|General|Forecasting|GPT-2
24 Mar 2024|[GPT4MTS: Prompt-Based Large Language Model for Multimodal Time-Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/30383)|University of Southern California|AAAI'24|Finance|Forecasting|GPT-2, BERT

### Vision

Date|Paper|Institute|Conference|Domain|Task|LLM
----|---------------------|----|----|----|----|----
26 Oct 2022|[IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text](https://arxiv.org/abs/2210.14395)|Meta|EMNLP'23 Findings|IoT|Classification, Retrieval|CLIP
17 Jan 2023|[Leveraging Vision-Language Models for Granular Market Change Prediction](https://arxiv.org/abs/2301.10166)|Johannes Kepler University Linz, Linz Institute of Technology|AAAI'23 Workshop|Finance|Forecasting|CLIP
4 May 2023|[Generating Virtual On-body Accelerometer Data from Virtual Textual Descriptions for Human Activity Recognition](https://arxiv.org/abs/2305.03187)|Georgia Institute of Technology|ISWC'23|IoT|Generation|ChatGPT
9 May 2023|[ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665)|Meta|CVPR'23|Audio, IoT|Classification, Retrieval|CLIP
25 May 2023|[PandaGPT: One Model To Instruction-Follow Them All](https://arxiv.org/abs/2305.16355)|University of Cambridge, Nara Institute of Science and Technology, etc.|Preprint|Audio, IoT|QA, Generation|ImageBind, Vicuna
27 Sep 2023|[AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model](https://arxiv.org/abs/2309.16058)|Meta|Preprint|Audio, IoT|Generation, QA, etc.|CLIP, LLaMA-2
27 Oct 2023|[Insight Miner: A Time Series Analysis Dataset for Cross-Domain Alignment with Natural Language](https://openreview.net/forum?id=E1khscdUdH&referrer=%5Bthe%20profile%20of%20Ming%20Zheng%5D(%2Fprofile%3Fid%3D~Ming_Zheng2))|UC Berkeley, Mineral, etc.|NeurIPS'23 Workshop|General|Generation|LLaVA, GPT-4

### Tool

Date|Paper|Institute|Conference|Domain|Task|LLM
----|---------------------|----|----|----|----|----
1 Jan 2023|[Unleashing the Power of Shared Label Structures for Human Activity Recognition](https://arxiv.org/abs/2301.03462)|UC San Diego, Amazon|CIKM'23|IoT|Classification|ImageBind, GPT-4
10 Jun 2023|[Language-Guided Traffic Simulation via Scene-Level Diffusion](https://arxiv.org/abs/2306.06344)|Columbia University, NVIDIA Research|CoRL'23|Traffic|Generation|GPT-4
31 Jul 2023|[ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)|Tsinghua University, ModelBest Inc., etc.|ICLR'24|General|Forecasting|LLaMA
9 Oct 2023|[Integrating Stock Features and Global Information via Large Language Models for Enhanced Stock Return Prediction](https://arxiv.org/abs/2310.05627)|Hithink RoyalFlush Information Network Co., Ltd.|IJCAI'23|Finance|Forecasting|LLaMA
30 Oct 2023|[GG-LLM: Geometrically Grounding Large Language Models for Zero-shot Human Activity Forecasting in Human-Aware Task Planning](https://arxiv.org/abs/2310.20034)|Samsung|Preprint|IoT|Forecasting|LLaMA
10 Feb 2024|[REALM: RAG-Driven Enhancement of Multimodal Electronic Health Records Analysis via Large Language Models](https://arxiv.org/abs/2402.07016)|Beihang University, China Mobile Research Institute|Preprint|Health|Classification|BERT, BGE-M3, Qwen
6 Mar 2024|[K-Link: Knowledge-Link Graph from LLMs for Enhanced Representation Learning in Multivariate Time-Series Data](https://arxiv.org/abs/2403.03645)|Institute for Infocomm Research, Nanyang Technological University|Preprint|General|Classification, Regression|CLIP

## Datasets

Dataset|Domain|Modality|Size|Task
-------|------|--------|----------|----
[Ego4D](https://ego4d-data.org/)|Audio, IoT|Text, IMU, Video, Audio, 3D|3, 670h data, 3.85M narrations|Classification, Forecasting, etc.
[Ego-Exo4D](https://ego-exo4d-data.org/)|Audio, IoT|Text, IMU, Video, Audio, Eye Gaze, Pose, 3D|1,422h data|Classification, Translation, etc.
[DeepSQA](https://github.com/nesl/DeepSQA)|IoT|Text, IMU|25h data, 91K questions|Classification, Question Answering
[PIXIU](https://github.com/chancefocus/PIXIU)|Finance|Text, Tables|136K instruction data|NER, QA, Forecasting, etc.
[MoAT](https://openreview.net/pdf?id=uRXxnoqDHH)|Finance, Health|Text, Time Series|6 datasets, 2K timesteps in total|Forecasting
[FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)|Finance|Text, Time Series|29.7M stock prices, 15.7M news records|Forecasting, Anomaly Detection, etc.
[Zuco 2.0](https://osf.io/2urht/)|Health|Text, Eye Tracking, EEG|739 sentences|Classification, Generation
[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)|Health|Text, ECG|60h data, 71 unique statements|Classification
[ECG-QA](https://github.com/Jwoo5/ecg-qa)|Health|Text, ECG|70 question templates|Classification, QA
[OpenAQA-5M](https://github.com/YuanGongND/ltu)|Audio|Text, Audio|5.6M (audio, question, answer) tuples|Tagging, Classification
[MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps)|Audio|Text, Music|5.5K music clips|Captioning, Generation
[CommonVoice](https://commonvoice.mozilla.org/en/datasets)|Audio|Text, Speech|7, 335 speech hours in 60 languages|ASR, Translation

## Citation

If you find this useful, please cite our paper: "Large Language Models for Time Series: A Survey"
```
@article{zhang2024large,
  title={Large Language Models for Time Series: A Survey},
  author={Zhang, Xiyuan and Chowdhury, Ranak Roy and Gupta, Rajesh K and Shang, Jingbo},
  journal={arXiv preprint arXiv:2402.01801},
  year={2024}
}
```
