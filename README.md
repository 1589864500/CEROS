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
Specifically, we define the indicator: $y_s=1$ if auditor confirms the claim's fraud risk based on the information in hospital $s$; otherwise $y_s=0$.
The fraud risk comfirmed by hospital set $S$ is $y_S=\max_{s\in S} y_s$, which is same as the formulation in InsComAllocation with monotonicity and submodularity properties.
The HosInvestigation dataset encompasses approximately one year's worth of data. Each claim typically involves around 6 candidate hospitals on average, with some claims considering up to 30 hospitals. This results in a dataset comprising 450,000 single-hospital samples and 7 million multi-hospital samples. 
In our models, claim features consist of the user's profile and claim-specific details, including the time interval between policy inception and claim submission.
Hospital features encompass their location, key departments, and historical comfirmation rate of claim fraud, among other relevant data.
