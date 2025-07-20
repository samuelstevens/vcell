# 06/29/2025

LambdaAI application

Related Publications (if any)
Provide a link to your own relevant papers, project pages, or other supporting materials such as demos or prototypes which may have shown good results on small datasets

BioCLIP: foundation vision model for all living organisms (university project, https://imageomics.github.io/bioclip/)
BioCLIP 2: sequel to BioCLIP, demonstrates that scale leads to emergent understanding of biological traits (university project, https://imageomics.github.io/bioclip-2/)
Mind the Gap and BioBench: evaluates both MLLMs and vision models on a wide spread of biology-relevant tasks with application-specific metrics (https://github.com/samuelstevens/mindthegap, https://samuelstevens.me/biobench)

Research Problem
What problem does your research address, why is it important, and how success looks like

Problem: Current AI models cannot reliably predict how cells respond to genetic perturbations across different cell types. Despite massive single-cell datasets, we lack standardized benchmarks to evaluate whether models capture generalizable biological mechanisms or just dataset artifacts.

Importance: Virtual cells that accurately predict perturbation responses would transform biology—replacing expensive experiments, accelerating drug discovery, and revealing causal gene-function relationships. The field currently wastes resources on models that fail to generalize beyond training contexts.

Success metrics: The Virtual Cell Challenge defines success through three complementary metrics:
1. Differential expression accuracy - predicting which genes change expression after perturbation
2. Perturbation discrimination - distinguishing between different genetic interventions by their effects  
3. Global expression prediction (MAE) - capturing the full transcriptomic response

Success means building models that excel across all three metrics when predicting responses to held-out perturbations in new cell types with minimal adaptation data. The immediate benchmark: accurately predicting 100 unseen genetic perturbations in H1 stem cells given only 150 training perturbations in that cell type, leveraging cross-cell-type knowledge from 350M+ cells in public datasets.

Long-term success: models that replace bench experiments for perturbation screening, enabling in silico exploration of genetic interventions across any human cell type.

Relevance and Novelty
What are the key related works and trends in this field, and how does your approach offer something new?


---

Based on this, it seems like we need an actual idea before getting compute grants.
