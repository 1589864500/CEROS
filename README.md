# CEROS: Cost-Efficient Fraud Risk Optimization with Submodularity in Insurance Claim

# Abstract
The cost-efficient fraud risk optimization with submodularity (CEROS) is a framework for insurance scenario. CEROS has been successfully applied in Alipay's healthcare insurance recommendation and claim verification fields, demonstrating potential for further adoption in numerous online resource allocation tasks. The CEROS framework encompasses two integral components: 

(a) The submodular set-wise classification model (SSCM), which 
models the fraud probability associated with multiple information sources and ensures the properties of submodularity of fraud risk 
without making independence assumption. Submodularity property, i.e., diminishing marginal return, often appears in the financial field, so SSCM considering the correlation between information sources is also useful in other financial fields.

(b) The primal-dual algorithm with segmentation point (PDA-SP), which exploits the submodularity of the objective function and its piecewise linear characteristics with respect to the dual components. It efficiently and theoretically identifies the optimal dual variables. These dual variables are derived from the Lagrangian dual method, serving as the weight parameters in the online decision-making stage.

# Real-World Datasets Details
We apply two real-world datasets from Alipay in our experiments, including InsComAllocation and HosInvestigation detailed as below.

(a) InsComAllocation. This dataset is associated with the scenario of insurer recommendation when users search for specific insurance products.
As insurers assess the viability of offering coverage based on their own risk control systems, it is essential for the platform to allocate a suitable set of insurers, enhancing the chance that users will receive insurance services.
A key component involves estimating the approval rate, influenced by the fraud likelihood as determined by the insurers' risk evaluation.
Specifically, we define the approval indicator: $y_s=1$ if insurer $s$ offers insurance to the user; otherwise, $y_s=0$.
The aggregate approval indicator for a set of insures $S$ is $y_S=\max_{s\in S} y_s$, showcasing both monotonicity and submodularity properties.
The InsComAllocation dataset encapsulates records from a three-month period.
On average, users are associated with about 2 candidate insurers, with a maximum of 13, yielding roughly 7 million samples for single-insurer scenarios and 15 million for multi-insurer contexts.
For modeling, the user features include basic profile information, credit ratings and statistical data, while insurers are represented solely by their ID.

(b) HosInvestigation. This dataset relates to the scenario of claim investigation when users submit a insurance claim. The platform has to allocate a set of hospitals for each claim, and auditors examine the medical records from each allocated hospital to ascertain if there are undisclosed pre-existing conditions associated with the claim. Therefore, a optimal set of hospitals that maximizes the detection of fraudulent claims while adhering to some global constraints is desired. This involves estimating the fraud probability as confirmed by multiple hospitals and then allocating an appropriate set of hospitals for a thorough investigation. 
Specifically, we define the indicator: $y_s=1$ if auditor confirms the claim's fraud risk based on the information in hospital 
$s$;
otherwise $y_s=0$.
The fraud risk comfirmed by hospital set $S$ is $y_S=\max_{s\in S} y_s$, which is same as the formulation in InsComAllocation with monotonicity and submodularity properties.
The HosInvestigation dataset encompasses approximately one year's worth of data. Each claim typically involves around 6 candidate hospitals on average, with some claims considering up to 30 hospitals. This results in a dataset comprising 450,000 single-hospital samples and 7 million multi-hospital samples. 
In our models, claim features consist of the user's profile and claim-specific details, including the time interval between policy inception and claim submission.
Hospital features encompass their location, key departments, and historical comfirmation rate of claim fraud, among other relevant data.

# Project Structure
```
CEROS
├─ SSCM  # The fraud probability estimator within the CEROS framework. SSCM captures the diminishing marginal return characteristic common in finance.
│  ├─ Baseline_NN_model.py  # SSCM的Baseline，DRSA-Net和SSCM_Ind.
│  ├─ SSCM_v1.py  # The proposed SSCM, complex feature processing version.
│  └─ SSCM_v2.py  # The proposed SSCM, simplified feature processing version.
└─ PDA-SP  # Dual variable optimizer for online decision-making within the CEROS framework. PDA-SP efficiently and theoretically searches the optimal dual parameters.
   ├─ algs  # The submodular optimization algorithms.
   │  ├─ greedy_general.py # Weak submodular optimization algorithm for PDA-SP.
   │  └─ guessK_greedy_general.py  # Strong submodular optimization algorithm [4] for PDA-SP.
   ├─ funcs  # Essential functions.
   │  ├─ funcs_mine.py  # File read/write operations.
   │  ├─ key_funcs_notorch.py  # Necessary functions for PDA-SP.
   │  ├─ key_funcs.py  # Necessary functions for PDA.
   │  └─ normal_funcs.py
   ├─ PDA-SP.py  # The proposed method for dual variables updating.
   └─ PDA.py  # Baseline, includes PDA-Adam, PDA-Adam-lrDecay, and PDA-Adam-GRS
```


# Method Implementations
## Outline
Table 1 Differences between SSCM and comparison methods
| Techs & Meths | SSCM | SSCM_Ind | DRSA-Net | lightGBM |
|---------|--------|----------|------------------|--------------|
| Sample Type | Set-wise | Point-wise | Point-wise | Point-wise |
| Submodularity | &#10003;   | &#10003; |  &#10003;  | &#10003; |
| Correlation   | &#10003;   | &#10007; |  &#10007;  | &#10007; |

*The submodularity of SSCM_Ind, DRSA-Net and lightGBM rely on the single-item modeling and independence assumption, which is an important contribution of CEROS, by capturing inter-project dependencies.*

*Modeling submodularity/diminishing marginal return is grounded in reality, a common scenario in the financial domain. For instance, when adding a new hospital $t$ to two combinations of hospitals, $A$ and $B$ ($A\subseteq B$), since the overlap between $B$ and $t$ is at least not less than that between $A$ and $t$, the increase in fraud probability that $t$ brings to $B$ will always be less than the increase it brings to $A$.*

*Modeling correlation among information resources is also grounded in reality, a facet previously overlooked in past work [1]. For instance, consider two identical hospitals, each with a fraud probability of 0.5 for claim investigation. After accounting for correlation, the combined fraud probability of the claim remains 0.5, contrary to the 0.75 predicted by the independence assumption. Especially in the insurance domain, where hospitals are limited yet significant, it is crucial to fully explore the information within the data.*


Table 2 Differences between PDA-SP and comparison methods
| Techs & Meths | PDA-SP | PDA-Adam | PDA-Adam-lrDecay | PDA-Adam-GRS |
|---------|--------|----------|------------------|--------------|
| Parameter Search Method  | Gradient + Analytical Expression   | Gradient |   Gradient  | Gradient |
| Submodularity | &#10003;   | &#10003; |  &#10003;   | &#10007; |
| Piecewise Linear Characteristics  | &#10003;   | &#10007; |   &#10007;  | &#10007; |
## SSCM
The objective of SSCM is to estimate the monotone submodular probability for a item set, such as the insurers or hospitals.
Unlike models that rely on the single-item modeling and independence assumption, SSCM captures the correlations among items with a data-driven manner. SSCM consists of three layers: point-wise encoding layer, set-wise aggregation layer and fraud probability estimation layer.
Specifically, the point-wise encoding layer encodes the features of individual items in the set, which can apply MLP or some effective network architecture for tabular data [2].
The set-wise aggregation layer explores the correlation between different items, followed by fraud probability estimation layer for prediction.
Each layer is designed to ensure the submodularity and monotonicity with some specific constraints.
And the following loss function is introduced to facilitate the learning process:
$l(p,y) = -((1+y) \log(1+p) + (1-y) \log(1-p) ) + y \log4\$.

## Baseline of SSCM
The baselines of SSCM employ the point-wise models that assume the independence of items for the set-wise classification task. Specifically, the model is trained using individual item samples, and the set-wise probability is computed under the independence assumption, neglecting any inter-item correlations.
Existing set-wise classification, such as aggregation with attention mechanism, can not satisfy the submodularity and monotonicity constraints.

### SSCM_Ind
SSCM_Ind has the same architechture as SSCM, except that it is trained on the point-wise samples. DRSA-Net, a novel deep tabular data modeling network, is adopted as the feature extrator in point-wise encoding layer of SSCM. The SSCM_Ind baseline aims to evaluate the correlation learning effectiveness of SSCM in set-wise fraud probability estimation task.

### DRSA-Net (MLP)
MLP shares the similar network structure with SSCM and SSCM_Ind, which also applies DRSA-Net as the feature extrator. It differs in three following aspects: 1) it uses the Relu function instead of the specific activation $\Phi(z)=\mathrm{log}(1+z)$; 2) it does not enforce weight non-negativity in the fraud probability estimation layer; 3) it employs sigmoid projection with cross entropy for training.
These variations aims to analyze the influence of submodularity-specific constraints in our model.

### Tree (LightGBM)
Tree uses the LightGBM library to train a GBDT model on the point-wise samples for binary classification. It is notable for its interpretability and is popular in financial scenarios. This LightGBM incorporates extensive prior knowledge observed from the data and complex data preprocessing. Indeed, LightGBM is competent for numerous prediction tasks in the fintech sector and was the model deployed in the previous generation by Alipay.

# PAD-SP
The objective of PDA-SP is to search the optimal dual variables (weights) for the online decision-making stage for the upcoming day from historical data. PDA-SP takes as input the historical dataset of the past 7 days, and its output is the optimal dual variables derived from planning on historical data. To achieve rapid convergence, PDA-SP incorporates two improvements over the general PDA approach: leveraging the submodularity of the objective function and exploiting the piecewise linear characteristics of the objective function with respect to the components of the dual variables.

The submodular optimization algorithms are implemented in `greedy_general.py` and `guessK_greedy_general.py`. The former implements a naive greedy algorithm starting from an empty set with a sample complexity of $O(N^2)$, while the latter implements an enumerate/guess-K greedy algorithm starting from enumerating sets of size $K=3$, performing simple greedy steps $C_N^3$ times, with a sample complexity reduced to approximately $O(N^4)$ through engineering optimizations [4]. Both methods theoretically provide a lower bound guarantee of 1-1/$\epsilon$ for most problems with simple constraints [3]. The guess-K greedy algorithm is recommended, as it is guaranteed to be no worse than the naive greedy algorithm.

The piecewise linear characteristics of the objective function with respect to the components of the dual variables are primarily manifested when the variables $\xi_{-k}$, excluding the dual variable component $\xi_k$ to be updated, are fixed. The objective function $\mathcal{L}(S_u|\xi)= F(S_u|u)-\sum_{k=1}^{M}\xi_kG_k(S_u|u)$ (Equation (8) in the paper) transforms into a piecewise linear function with respect to $\xi_k$, where both $G(\cdot)$ and $F(\cdot)$ become constant terms. This transformation is demonstrated in L#506-514 of the paper:
$\mathcal{L}(\xi_k)= f(\xi_k|u,S^t_u,\xi^t_{-k})=-G_k(S^t_u|u)\xi_k-\sum_{k'=1,k'\neq k}^M\xi^t_{k'}G_{k'}(S^t_u|u)+F(S^t_u|u)$.
Analyzing this formula allows for the determination of the nearest left and right segment endpoints in the domain relative to the variables $\mathbf{\xi}_{-k}$ to be updated. Updating directly at these endpoints significantly enhances the efficiency of updates. Furthermore, Theorem 4.2 in the paper provides robust theoretical support under the assumption of monotonic submodularity.

## Baseline of PAD-SP
### PDA-Adam
PDA-Adam uses Adam optimizer to search the dual variables with primal-dual framework, PDA-Adam follows configurations from the resource allocation framework under video homepage generation scenario [1] with a fixed learning rate and L2 regularization term.
### PDA-Adam-lrDecay
PDA-Adam-lrDecay is designed to enhance PDA-Adam's performance. PDA-Adam-lrDecay uses Adam optimizer with a learning rate variable-step multi-stage decay operator. Specifically, PDA-Adam-lrDecay with refined hyperparameters adopts a 3-phase decay approach. PDA-Adam-lrDecay implements a learning rate decay with 5 refined hyperparameters: [0,100] with the learning rate of 0.12, [100,300] with the learning rate of 0.08, [300,1000] with the learning rate of 0.04. The hyperparameters in PDA-Adam-lrDecay are optimized through numerous repeated experiments to ensure the best convergence efficiency in offline healthcare insurance claim problems. 
Additionally, instead of choosing exponential decay or fixed-step multi-stage decay operators, the choice of variable-step multi-stage decay operator is to prevent early stopping while maintaining flexibility. 
### PDA-Adam-GRS
PDA-Adam-GRS, similar to PDA-Adam, employs the Adam optimizer with a fixed learning rate and an L2 regularization term. However, distinct from PDA-Adam and PDA-Adam-lrDecay, PDA-Adam-GRS does not exploit the submodularity of the objective function. Instead, PDA-Adam-GRS adopts a Greedy Random Search (GRS) strategy, which bypasses the utilization of submodularity. The GRS approach initiates from an empty strategy and, over $|E|$ iterations, randomly selects non-duplicate hospitals from the set $E$, thereby generating a sequence of candidate strategies with sizes ranging from 1 to $|E|$. This method provides an alternative strategy without relying on the submodular properties of the objective function, aiming to demonstrate the importance of exploring the submodularity.


# Citation
[1] Weicong Ding, Dinesh Govindaraj, and SVN Vishwanathan. 2019. Whole page optimization with global constraints. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD). ACM, New York, New York, 3153–3161.

[2] Qinghua Zheng, Zhen Peng, Zhuohang Dang, Linchao Zhu, Ziqi Liu, Zhiqiang Zhang, and Jun Zhou. 2023. Deep Tabular Data Modeling With Dual-Route Structure-Adaptive Graph Networks. IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING 35, 9 (2023), 9715–9727

[3] Michael Kapralov, Ian Post, and Jan Vondrák. 2013. Online Submodular Welfare Maximization: Greedy is Optimal. In Proceedings of the 24th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA). SIAM, New Orleans, Louisiana, 1216–1225.

[4] Moran Feldman, Zeev Nutov, and Elad Shoham. 2023. Practical Budgeted Submodular Maximization. Algorithmica 85, 5 (2023), 1332–1371
