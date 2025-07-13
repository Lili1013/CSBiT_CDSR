# CSBiT_CDSR
# THis is the source code for the paper "Enhancing LLM-based Cross-Domain Sequential Recommendation with CLuster Sampling and Bi-Step Instruction Tuning"
# Please run the code based on the following order. 
# data_preprocess
## 1. construct_inter_data_from_review.py
## 2. process_meta_data.py
## 3. construct_inter_data_general.py; construct_inter_data_specific.py
## 4. item_embedding.py
## 5. generate_new_sequences.py
## 6. split_datasets_loo.py
## 7. construct_prompt_general.py; construct_prompt_specific.py
# source_code
## 1. cluster_sampling.py
## 2. train_llm_phase_1.py
## 3. cal_user_rep.py
## 4. train_llm_phase_2.py
## 5. inference_phase_2.py
## 6. evaluate_llm.py