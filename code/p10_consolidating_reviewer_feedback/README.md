# Consolidating Review feedback

## Reviewer 1 feedback

> **1. Suggest provide more details about the five-fold cross-validation procedure and data partitioning method.**
> 2. If possible suggest enhance discuss potential experimental challenges in implementing these simulated structures.
> **3. Suggest enhance explain why nonlinear models weren't explored for Si's nonlinear spectral responses.**
> **4. Suggest elaborate on the algorithmic process for identifying optimal single-wavelength predictors.**
> 5. Suggest if possbile expand discussion on polarization effects and their implications for real-world sensing scenarios.
> 6. Suggest the authors enhance the introdcution part about biosensor wperformance enhance with machine leraning, one literature are suggest:Wearable photonic smart wristband for cardiorespiratory function assessment and biometric identification, Machine Learning Enhances the Performance of Bioreceptor-Free Biosensors etc 

## Reviewer 2 feedback

> Major Problems:
> 
> **1. Rationale for 80 principal components in PCA: The authors use 80 principal components for linear regression but does not explain the basis for selecting "80" (e.g., cumulative variance contribution rate). Excessive principal components may introduce noise, while too few may lose key information. It is suggested to supplement the detailed PCA dimensionality reduction process to justify the necessity of 80 principal components.**
> **2. Details of five-fold cross-validation: The authors use five-fold cross-validation to evaluate the generalization ability of their model but does not clarify the data partitioning method (e.g., whether stratified sampling by refractive index range is used, or whether there is a risk of data leakage). If the refractive index distributions of the training and test sets differ significantly, the generalization ability of the model may be overestimated. It is suggested to supplement the specific cross-validation strategy.**
> **3. Incomplete justification of model selection: The study concludes that linear models are ill-suited for the nonlinear spectral response of Si-based sensors. This is a reasonable conclusion, but it remains an incomplete analysis because the logical next step is not taken. The authors should have applied a suitable nonlinear regression model (e.g., SVR with a nonlinear kernel, a small neural network, or gradient boosting) to the Si-based datasets. This would serve two purposes: a) It would definitively prove that the performance limitation lies with the model's linearity, not with the full-spectrum approach itself. b) It would provide a much fairer and more insightful comparison, showing how much better a nonlinear model performs on resonant spectra. The brief mention on page 11 that linear regression was chosen after testing other models on a single dataset (Ti-TE) is insufficient. The Ti-TE data is shown to be highly linear, so it is unsurprising that a linear model performed best. The critical test case is the nonlinear Si data, which was not evaluated with a nonlinear model. This is a major missed opportunity that weakens the paper's overall conclusion.**
> 4. Feasibility for practical applications: The research is based on simulated data and does not consider the impact of experimental noise (e.g., random noise in spectral measurements, temperature drift) on the model. Full-spectrum methods require collecting complete spectra, which imposes higher requirements on hardware (e.g., spectrometer resolution) compared to single-wavelength methods. It is recommended to discuss applicability in actual noisy experimental scenarios and potential optimization methods (e.g., noise reduction processing).
> 
> Minor Problem:
> 
> Page 3, Line 93: Typo in Figure 1 caption: "nanords" should be "nanorods."