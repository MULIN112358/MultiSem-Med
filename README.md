# MultiSem-Med: Multi-Source Medical Knowledge–Enhanced Medication Recommendation

## Process Data

1. Download the raw datasets from:

   • MIMIC-III: https://physionet.org/content/mimiciii/1.4/

   • MIMIC-IV: https://physionet.org/content/mimiciv/3.1/

   Note: Access to PhysioNet datasets requires credentialed authorization and acceptance of the data use agreement.
2. For a fair comparison, we adopt the same preprocessing pipeline as used
   in [Carmen](https://github.com/bit1029public/Carmen). Please refer to their repository for implementation details.

## Train Model

```
python main.py \
--Test False \
--model_name MultiSem-Med \
--ddi True \
--lr 1e-4 \
--target_ddi 0.06 \
--dim 64 \
--nhead 8 \
--cuda 0 \
--dataset mimic-iii
```

## Test Model

```
python main.py \
--Test True \
--model_name MultiSem-Med \
--resume_path Epoch_17_JA_0.5463_DDI_0.07331.model \
--ddi True \
--lr 1e-4 \
--target_ddi 0.06 \
--dim 64 \
--nhead 8 \
--cuda 0 \
--dataset mimic-iii
```