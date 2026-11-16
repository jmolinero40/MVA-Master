# The seed of ChatGPT and other Large Language Models: Reflections on "Improving Language Understanding by Generative Pre-Training"

## Javier Molinero Araguas

In “Improving Language Understanding by Generative Pre-Training”, Radford et al. (2018) show that a Transformer decoder, first trained as a language model on a large corpus of unlabeled text, can be adapted to a wide range of natural language processing tasks and outperform state-of-the-art models on most of them. Their approach relies on two stages: generative pre-training on BooksCorpus, followed by supervised fine-tuning with only minimal changes to the architecture. In this essay, I discuss the role of this paper in modern LLMs and its zero-shot behaviors, and I carry out an experiment on one of the open questions the authors left.

I find this paper particularly interesting because it is one of the seeds of modern LLMs. What strikes me the most is the great performance the authors obtain by using a single architecture for all tasks, instead of the task-specific architectures of state-of-the-art techniques such as LSTM-based methods. Moreover, the model brings a broad understanding of language through unsupervised methods and transfers it to multiple different tasks by simply adding input transformations. The extraction token `<e>` particularly drew my attention. Placed at the end of the sequence and due to the masked self-attention, it is the only token that sees the whole sentence. Therefore, it acts as the bridge between understanding and transferring knowledge.

What surprised me first is how each layer of the pre-trained model contains useful functionality and therefore improves performance. This is not a trivial deduction, since the experiment does not add capacity, but only changes how many layers start from pre-trained weights. Finally, I did not expect the zero-shot behaviors that the authors present. It is quite surprising how the language-modeling objective seems to lead the model to learn specific tasks such as sentiment analysis. As they show in Figure 2 (right), these abilities steadily improve during pre-training, which suggests that they could be an intrinsic aspect of language acquisition. This prompts me to question whether human learning in children functions the same way.

On the other hand, some open questions remain. The model still needs labeled data and fine-tuning for each task, and the authors only offer a hypothesis to explain the zero-shot behaviors, while a deeper explanation is something that I would have loved to see. Nonetheless, this paper is a milestone for NLP and AI in general.

Among the open questions the authors left, there is one quite interesting about smaller sets. On RTE (about 2,500 training examples), GPT-1 obtains 56.0% accuracy, below the 61.7% of a multi-task BiLSTM. The authors stated that their model would likely benefit from multi-task training as well, but they did not explore it. This uncertainty piqued my interest because of the connections with the way human beings learn. Let us imagine a child who has learnt about textual entailment through 20,000 examples of novels and common conversations. Would he be able to distinguish entailment in news? What if we taught him all together, with news representing a minority of the examples? Is entailment a universal capability or a domain-specific one? Can we add domain-specific knowledge (like news) to a person who is already an expert in entailment in other domains?

To try to elucidate this, I downloaded GPT-1, MNLI and RTE from Hugging Face and fine-tuned four models, all evaluated on RTE:

A. **RTE-only.** The baseline shown in the paper.

B. **MNLI-only.** Is entailment learned in other domains transferred to news without seeing any of them?

C. **MNLI → RTE.** Fine-tuned on MNLI, later on RTE, as in STILTs (Phang et al., 2018). Can we add domain-specific knowledge to a model that is already an expert in entailment?

D. **MNLI + RTE.** A form of multi-task training, although not in its classical form, which uses one classification head per task instead of a single shared one. I multiplied the RTE set by three, following the temperature-scaled mixing of Raffel et al. (2020) (calculations in the notebook).

I kept the fine-tuning hyperparameters reported by the authors and used a balanced and binarized subset of 20,000 MNLI examples. Each model is trained with 5 seeds and, for simplicity, without the auxiliary language-modelling loss, so I compare the baseline with the one in Table 5 of the paper. All models are evaluated on the RTE validation set (277 examples), since the test labels are hidden, and their significance is assessed with statistical tests (Table 2). Further details are in the notebook.

**Table 1: RTE validation accuracy (5 seeds). True entailment proportion of 52.7%.**

|Condition|Accuracy (%)|Entailment pred. (%)|
|:-|:-:|:-:|
|A · RTE|55.09 ± 3.44|61.37|
|B · MNLI|64.12 ± 0.91|69.24|
|C · MNLI → RTE|66.71 ± 1.50|61.59|
|D · MNLI + RTE×3|67.00 ± 1.87|56.68|

**Table 2: Statistical tests (paired test only for B–C, since each C run starts from the B model with the same seed).**

|Pair|Diff. (pts)|p (Welch)|p (paired)|
|:-|:-:|:-:|:-:|
|A–D|11.91|0.0004|-|
|B–C|2.60|0.0139|0.0086|
|C–D|0.29|0.7945|-|

The results of the experiment confirm the authors’ hypothesis. Indeed, multi-task training with a larger set (MNLI) performs better on a smaller set than training only on this one (A vs D +11.91 points). Moreover, models trained with MNLI surpass the multi-task BiLSTM (67% vs 61.7%), although our results are measured on the validation set and theirs on the test set. In addition, my baseline is consistent with the authors’ result (55.09% ± 3.44 vs 54.4%).

Interestingly, MNLI-only outperforms RTE-only by 9 points, so language entailment can indeed be transferred from bigger datasets to domain-specific smaller ones. Nevertheless, introducing the smaller set after MNLI is still useful (B vs C +2.6 points, statistically significant).

Additionally, the way of introducing the smaller set does not seem to matter, since there is no statistical evidence that sequential and multi-task training are different (C vs D).

The baseline also shows a certain instability (std of 3.44 points), which the introduction of MNLI seems to reduce (0.91-1.87 points). We might also remark a certain bias towards the entailment class (57-69% of predictions vs 52.7% of true labels), strongest in MNLI-only.

This pedagogical exercise is limited by the capacity. We are only using 5 seeds, a subset of MNLI, a simple loss function and a simplified multi-task model. It has however been useful to answer an open question in the paper, while teaching me a lot of concepts and techniques in the process.

In conclusion, the experiment illustrates that the pre-trained model transfers its knowledge very well, but it still needs enough labeled examples to unlock a task, and a small dataset like RTE is not enough on its own. Going back to the child I imagined before, he would indeed recognize entailment in the news after learning it from novels and conversations, and a few news examples would only help him a bit more. Finally, this dependence on labeled data is precisely what GPT-2 and GPT-3 started to reduce, which shows why this paper can be considered the seed of modern large language models.

## References

Phang, J., Févry, T., & Bowman, S. R. (2018). Sentence encoders on STILTs: Supplementary training on intermediate labeled-data tasks. arXiv:1811.01088.

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training. OpenAI.

Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR.

Datasets: RTE (Bentivogli et al., 2009), MNLI (Williams et al., 2018), via GLUE (Wang et al., 2018).
