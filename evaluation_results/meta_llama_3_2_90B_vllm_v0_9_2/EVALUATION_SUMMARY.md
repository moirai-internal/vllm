# Comprehensive Evaluation Summary

**Timestamp:** Thu Aug 21 17:39:24 UTC 2025
**Model:** vllm-model
**Server:** http://localhost:8080

## Test Results

### 1. Feature Tests
- **Status:** ✅ Completed
- **Log:** feature_tests.log

#### Feature Test Summary (last 10 lines):
```
2025-08-21 14:52:45,799 - INFO - ============================================================
2025-08-21 14:52:45,799 - INFO - Total tests: 4
2025-08-21 14:52:45,799 - INFO - Passed tests: 4
2025-08-21 14:52:45,799 - INFO - Failed tests: 0
2025-08-21 14:52:45,799 - INFO - Pass rate: 100.00%
2025-08-21 14:52:45,799 - INFO - All feature sanity checks completed successfully.
2025-08-21 14:52:45,799 - INFO - ============================================================
Configuration: {'server_engine': 'vLLM', 'server_version': 'v0.9.2.881ad1a', 'server_api': 'http://localhost:8080/v1/chat/completions', 'model_name': 'vllm-model'}
```

### 2. Version Comparison
- **Status:** ✅ Completed
- **Results:** version_comparison.csv
- **Log:** version_comparison.log

#### Version Comparison Details:
```
    Position 4: 4 times (13.3%)
    Position 66: 3 times (10.0%)
    Position 92: 3 times (10.0%)
    Position 84: 3 times (10.0%)
    Position 37: 3 times (10.0%)

  Difference types:
    token_mismatch: 30 (100.0%)

  Difference timing:
    Early differences (pos < 10): 6 (20.0%)
    Late differences (pos >= 10): 24 (80.0%)

============================================================
COMPARISON SUMMARY
============================================================
  Current Version: v0.9.2.881ad1a
  Baseline Version: v0.6.4.post1.0c9082a1
  Total tests: 33
  Mismatches: 30
  Missing baseline: 0
  Detailed analysis saved to: /home/vasheno/sanity_check/evaluation_results_vLLM_v0_9_2_881ad1a_20250821_145240/version_comparison.csv

Done!
```

### 3. Consistency Checks
- **Status:** ✅ Completed
- **Log:** consistency_check.log

#### Consistency Check Summary (last 8 lines):
```
2025-08-21 15:06:35,243 - INFO - ============================================================
2025-08-21 15:06:35,243 - INFO - CONSISTENCY CHECK SUMMARY
2025-08-21 15:06:35,243 - INFO - ============================================================
2025-08-21 15:06:35,243 - INFO - Sequential tests average consistency rate: 100.00%
2025-08-21 15:06:35,243 - INFO - Concurrent tests average consistency rate: 46.09%
2025-08-21 15:06:35,243 - INFO - Overall average consistency rate: 73.05%
2025-08-21 15:06:35,243 - INFO - ============================================================
Configuration: {'server_engine': 'vLLM', 'server_version': 'v0.9.2.881ad1a', 'server_api': 'http://localhost:8080/v1/chat/completions', 'model_name': 'vllm-model'}
```

### 4. lm_eval (MMLU Pro)
- **Status:** ✅ Completed
- **Results:** lm_eval_results/
- **Log:** lm_eval.log

#### lm_eval Results (last 5 lines):
```

| Groups |Version|    Filter    |n-shot|  Metric   |   |Value |   |Stderr|
|--------|------:|--------------|------|-----------|---|-----:|---|-----:|
|mmlu_pro|      2|custom-extract|      |exact_match|↑  |0.6792|±  |0.0042|

```

### 5. BFCL (Function Calling)
- **Status:** ✅ Completed
- **Results:** Check BFCL_PROJECT_ROOT: /home/vasheno/sanity_check/evaluation_results_vLLM_v0_9_2_881ad1a_20250821_145240/bfcl_results
- **Logs:** bfcl.log, bfcl_eval.log

#### BFCL Overall Accuracy:
```
Rank,Overall Acc
1,19.73%
```

### 6. LooGLE (Long Document QA)
- **Status:** ✅ Completed
- **Results:** loogle_results/
- **Log:** loogle.log

#### LooGLE Results (last 2 lines):
```
Scoring batches:   0%|          | 0/18 [00:00<?, ?it/s]Scoring batches:   6%|▌         | 1/18 [00:06<01:44,  6.15s/it]Scoring batches:  11%|█         | 2/18 [00:13<01:46,  6.63s/it]Scoring batches:  17%|█▋        | 3/18 [00:17<01:25,  5.73s/it]Scoring batches:  22%|██▏       | 4/18 [00:23<01:18,  5.58s/it]Scoring batches:  28%|██▊       | 5/18 [00:29<01:17,  5.95s/it]Scoring batches:  33%|███▎      | 6/18 [00:37<01:18,  6.55s/it]Scoring batches:  39%|███▉      | 7/18 [00:42<01:07,  6.09s/it]Scoring batches:  44%|████▍     | 8/18 [00:46<00:55,  5.54s/it]Scoring batches:  50%|█████     | 9/18 [00:49<00:41,  4.61s/it]Scoring batches:  56%|█████▌    | 10/18 [00:52<00:32,  4.05s/it]Scoring batches:  61%|██████    | 11/18 [00:56<00:28,  4.06s/it]Scoring batches:  67%|██████▋   | 12/18 [00:58<00:20,  3.46s/it]Scoring batches:  72%|███████▏  | 13/18 [01:02<00:17,  3.59s/it]Scoring batches:  78%|███████▊  | 14/18 [01:06<00:15,  3.89s/it]Scoring batches:  83%|████████▎ | 15/18 [01:09<00:10,  3.51s/it]Scoring batches:  89%|████████▉ | 16/18 [01:12<00:06,  3.38s/it]Scoring batches:  94%|█████████▍| 17/18 [01:19<00:04,  4.53s/it]Scoring batches: 100%|██████████| 18/18 [01:21<00:00,  3.53s/it]Scoring batches: 100%|██████████| 18/18 [01:21<00:00,  4.50s/it]
Average BERTScore (F1): 86.23%
```

## Files Generated
- evaluation.log
- feature_tests.log
- bfcl_results/score/data_multi_turn.csv
- bfcl_results/score/data_live.csv
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multi_turn_long_context_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_irrelevance_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_simple_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_javascript_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_parallel_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multi_turn_miss_param_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multi_turn_miss_func_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_simple_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_parallel_multiple_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_relevance_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_parallel_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_multiple_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multi_turn_base_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_parallel_multiple_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multiple_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_java_score.json
- bfcl_results/score/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_irrelevance_score.json
- bfcl_results/score/data_non_live.csv
- bfcl_results/score/data_overall.csv
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_irrelevance_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_javascript_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_simple_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multi_turn_long_context_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_java_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multiple_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_relevance_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multi_turn_miss_param_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multi_turn_base_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_parallel_multiple_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_parallel_multiple_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_simple_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_parallel_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_multi_turn_miss_func_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_parallel_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_irrelevance_result.json
- bfcl_results/result/meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC/BFCL_v3_live_multiple_result.json
- lm_eval_results/vllm-model/results_2025-08-21T16-52-05.656496.json
- version_comparison.log
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_010_52053bfb_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_002_a5985e07_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_007_9151a846_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_000_4491cff4_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_003_1b24647b_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_001_72261cc0_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_006_f4dd7e8d_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_005_2ae6d653_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_009_90e12500_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_004_78fe0856_results.json
- vllm-model_vLLM_v0_6_4_post1_0c9082a1/prompt_008_82f83a61_results.json
- EVALUATION_SUMMARY.md
- consistency_check.log
- loogle.log
- lm_eval.log
- bfcl.log
- bfcl_eval.log
- version_comparison.csv
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_010_52053bfb_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_002_a5985e07_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_007_9151a846_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_000_4491cff4_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_003_1b24647b_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_001_72261cc0_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_006_f4dd7e8d_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_005_2ae6d653_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_009_90e12500_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_004_78fe0856_results.json
- vllm-model_vLLM_v0_9_2_881ad1a/prompt_008_82f83a61_results.json
