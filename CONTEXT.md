# Passion Fruit Quality Grading

This context describes how fixed-view observations of passion fruits become decision-layer samples and quality grades.

## Language

**Fruit**:
A physical passion fruit photographed once from each of the three fixed views.
_Avoid_: Sample, image

**View Observation**:
One photo of one Fruit from a fixed view. It is one independent decision-layer sample and one CSV row.
_Avoid_: Fruit sample, image ID

**View ID**:
The fixed capture position of a View Observation, identified as `1`, `2`, or `3`. Together, Fruit ID and View ID uniquely identify a View Observation.
_Avoid_: Image ID

**Image Grade**:
The expert-assigned ordinal quality grade of one View Observation: `0` for `加工`, `1` for `下等`, `2` for `中等`, and `3` for `上等`.
_Avoid_: Folder grade, fruit grade

**Fruit Grade**:
The final quality grade of a Fruit, defined as the worst of its three Image Grades.
_Avoid_: Image grade

**Legacy Fruit Grade**:
The original whole-fruit grade used to organize the collected image folders. It is not an Image Grade and is not a decision-layer training label.
_Avoid_: Grade

**Fruit Area Ratio**:
The fruit bounding-box area divided by the original image area under the fixed capture setup. It is a relative size proxy, not a physical area measurement.
_Avoid_: Fruit size, physical area

**Defect Area Ratio**:
The visible union area of one defect class divided by the visible fruit ROI area in one View Observation.
_Avoid_: Defect pixel sum, defect area

**Invalid View Observation**:
A View Observation whose ROI extraction, model inference, or mask calculation did not complete successfully. It is retained for review but excluded from decision-layer training until corrected.
_Avoid_: No defect, zero defect

**Decision Dataset**:
The canonical collection of View Observations used by the decision layer, partitioned into `train`, `valid`, and `test` without splitting a Fruit across partitions.
_Avoid_: Training CSV, split CSV

**Model Bundle**:
A versioned set of ROI, color, wrinkle, and local-defect models used together to derive one Decision Dataset's features.
_Avoid_: Model, weight file

**Model Manifest**:
The YAML record that binds every Model Bundle role to a weight hash, task, architecture, image size, class mapping, threshold, file size, and approval status. A Decision Dataset version is frozen to the exact Model Manifest hash.
_Avoid_: Model settings, weight list

**Dataset Version**:
An immutable output namespace such as `v001`. Replacing a model requires a new Dataset Version. A changed source image may replace its previous observation only through the explicit replacement flow, which clears its Image Grade.
_Avoid_: Run folder, CSV name

**Complete Fruit**:
A Fruit with all three required View Observations present and valid. Only Complete Fruits participate in fruit-level validation and test evaluation.
_Avoid_: Valid image, complete row

**Dangerous Overgrade**:
A prediction whose grade is numerically higher than the expert-assigned Image Grade or Fruit Grade, causing quality to be overstated.
_Avoid_: False positive, ordinary misclassification
