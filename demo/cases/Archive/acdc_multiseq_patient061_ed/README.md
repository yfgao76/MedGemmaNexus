# Cardiac Multi-Sequence Demo Case (patient061 ED)

This case bundles both:
- Cine: `patient061_4d.nii.gz`
- MOLLI-T1 map (ED): `patient061_t1_molli_map.nii.gz`

The folder is symlink-based and points to:
- `MRI_Agent/demo/cases/acdc_test_group_4_per_group/patients/patient061`
- `MRI_Agent/demo/cases/acdc_test_group_4_per_group/derived_t1_molli/t1_molli_nifti/patient061_ed_t1_molli_map.nii.gz`

## 1) Agent mode (cardiac, multi-sequence)

```bash
cd /home/longz2/common/medgemma
python -m MRI_Agent.agent.loop \
  --case-id acdc_multiseq_patient061_ed \
  --dicom-case-dir /home/longz2/common/medgemma/MRI_Agent/demo/cases/acdc_multiseq_patient061_ed \
  --domain cardiac \
  --llm-mode server \
  --server-base-url http://127.0.0.1:8000 \
  --server-model <YOUR_VLM_MODEL_ID> \
  --autofix-mode force \
  --max-steps 12 \
  --runs-root MRI_Agent/runs
```

`identify_sequences` should map both `CINE` and `T1`.

## 2) Alignment check (ED frame vs T1)

`alignment_check_ed_t1_vs_frame01.json` is pre-generated in this folder.
Current result: `same_grid = true`.

## 3) T1 feature pyramid + rule-based diagnosis (with mask labels)

Example using ACDC ED GT labels (1=RV, 2=MYO, 3=LV):

```bash
cd /home/longz2/common/medgemma
python MRI_Agent/scripts/cardiac_t1_pyramid_diagnosis.py \
  --t1-map MRI_Agent/demo/cases/acdc_multiseq_patient061_ed/patient061_t1_molli_map.nii.gz \
  --label-seg MRI_Agent/demo/cases/acdc_test_group_4_per_group/patients/patient061/patient061_frame01_gt.nii.gz \
  --subject-id patient061 \
  --phase ED \
  --output-dir MRI_Agent/demo/cases/acdc_multiseq_patient061_ed/t1_pyramid_output
```

Outputs:
- `t1_pyramid_diagnosis.json`
- `t1_slice_features.csv`
- `t1_sector_pyramid_features.csv`

