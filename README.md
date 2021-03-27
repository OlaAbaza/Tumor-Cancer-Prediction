## Project: Tumor Cancer Prediction.

The objective of the projects is to prepare you to apply different machine learning algorithms to real-world tasks. This will help you to increase your knowledge about the workflow of the machine learning tasks. You will learn how to clean your data, applying pre-processing, feature engineering, classification methods.

# **- preprocessing techniques.**

- Check for missing value. 


<img src="images/11.PNG"/>
<img src="images/null.PNG"/>

- map the class label

Transform the class labels from their original string representation (M and B) into integers

<p>
<img src="images/map.PNG"/>
</p>

- Feature Standardization.

Use sklearn to scale and transform the data

![](RackMultipart20210327-4-12nzwsu_html_355d85e8ce8e528a.png)

# **-Data analysis.**

- **DataFrame .describe()**

Calculating some statistical data like  **percentile, mean**  and  **std**  of the numerical values of the Series or DataFrame.

![](RackMultipart20210327-4-12nzwsu_html_6828bcae8f66e2d0.png)

-
# Correlation Matrix .

![](RackMultipart20210327-4-12nzwsu_html_aa11bffd4434992c.png)

### Observation:

- The f2 and f4 feature have a strong positive correlation with f6,f8 and f9 feature;
- The f21 and f22 feature have a weak correlation with f24,f8 and f19 feature;

- countplot

### Observation:

Number of benign tumor data more than number of malignant tumor data

![](RackMultipart20210327-4-12nzwsu_html_4f0d19e38e221f72.png)

- boxplot

### Observation:

mostof the values are usually higher in malignant than that of benign ![](RackMultipart20210327-4-12nzwsu_html_7354848c7cc1ef61.png) ![](RackMultipart20210327-4-12nzwsu_html_8ab7aa6deab93ac0.png) ![](RackMultipart20210327-4-12nzwsu_html_61a9cc04c270160a.png) ![](RackMultipart20210327-4-12nzwsu_html_f4f7262f61c2135.png)

# **-**

#

# **Sizes of training and validation sets.**

80% of the data for training and the remaining 20% for validation.

![](RackMultipart20210327-4-12nzwsu_html_ad1bf1206edcc79b.png)

# **- Hyperparameter tuning.**

- **SVM**

Hyperparameter:

- Kernel:

- sigmoid

accuracy score =0.945054945054945.

- linear

accuracy score =0.978021978021978.

- rbf

accuracy score =0.978021978021978.

- gamma:

- 0.001

accuracy score =0.9560439560439561.

- 0.0001

accuracy score =0.7362637362637363.

- 0.01

Accuracy score =0.978021978021978.

- **Decision Tree.**

Hyperparameter:

- max\_depth:

- (None)

Accuracy score =0.9340659340659341.

- (2**)**

Accuracy score =0.9560439560439561.

- (4**)**

Accuracy score =0.945054945054945

- min\_samples\_leaf:

- (10)

accuracy score =0.9560439560439561.

- (6**)**

Accuracy score =0.967032967032967

- (4**)**

Accuracy score =0.9340659340659341

- **xgboost.**

Hyperparameter:

- max\_depth:

- (3)

accuracy score =0.978021978021978

- (2**)**

Accuracy score =0.967032967032967

- (4**)**

Accuracy score =0.967032967032967

- learning\_rate:

- (0.05)

Accuracy score =0.978021978021978.

- (0.5**)**

Accuracy score =0.967032967032967

- (0.10**)**

Accuracy score =0.978021978021978

# **- Dimensionality Reduction.**

- **PCA**

- SVM:

- (0.90)

Accuracy score =0.978021978021978.

- (0.50**)**

Accuracy score =0.9340659340659341

- (25**)**

Accuracy score =0.978021978021978

.

- Decision Tree:

- (0.90)

Accuracy score =0.9340659340659341

- (25**)**

Accuracy score =0.9340659340659341

- (0.70**)**

Accuracy score =0.9230769230769231

.

- xgboost:

- (0.90)

Accuracy score =0.945054945054945

- (0.50**)**

Accuracy score =0.9340659340659341

- (24**)**

Accuracy score =0.9560439560439561

.

.

# **- Training Time graph.**

![](RackMultipart20210327-4-12nzwsu_html_4a3086e2c68a30a5.png)

# **- Testing Time graph.**

![](RackMultipart20210327-4-12nzwsu_html_65a98d0d65b53a18.png)

# **- Testing Time graph with using PCA.**

![](RackMultipart20210327-4-12nzwsu_html_4c8563d6d13da1fb.png)

# **- Training Time graph with using PCA.**

![](RackMultipart20210327-4-12nzwsu_html_b01d9605402332ae.png)

# **- Accuracy graph**

![](RackMultipart20210327-4-12nzwsu_html_4360809b8b0d1c45.png)

# **- Accuracy graph with using PCA**

![](RackMultipart20210327-4-12nzwsu_html_91c9e0dd49a639f7.png)

##
# **Summary**

_We applied_ _Decision Tree,__XGBoosts_ _and Support Vector Machine (SVM)_

_algorithms __to__ the Tumor Cancer dataset._

• _To predict whether the Tumor cancer is malignant or benign._
_• Compared the performance results of all the algorithms based on_

_the accuracy values. __and showed that_ _XGBoosts__ classifier is the best_

_among all in determining benign and malignant tumors._