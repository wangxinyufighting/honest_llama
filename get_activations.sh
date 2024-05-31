python get_activations.py llama2_chat_7B gsm8k_train_test --device 0 
# python get_activations.py vicuna_7B hotpot_qatest_medium_has_support_vicuna-7b_1_smaple --device 0 n


# python get_activations.py llama2_chat_7B hotpot_qatrain_hard_has_support_vicuna-13b_1_sample_equals --device 0 
# python get_activations.py honest_llama_7B tqa_gen_end_q --device 0 
# CUDA_VISIBLE_DEVICES=0 python get_activations.py alpaca_7B tqa_mc2 --device 0 
# CUDA_VISIBLE_DEVICES=0 python get_activations.py alpaca_7B tqa_gen_end_q --device 0 
# CUDA_VISIBLE_DEVICES=0 python get_activations.py vicuna_7B tqa_mc2 --device 0 
# CUDA_VISIBLE_DEVICES=0 python get_activations.py vicuna_7B tqa_gen_end_q --device 0 

# CUDA_VISIBLE_DEVICES=0 python get_activations.py llama2_chat_7B tqa_mc2 --device 0
# CUDA_VISIBLE_DEVICES=0 python get_activations.py llama2_chat_7B tqa_gen_end_q --device 0

# CUDA_VISIBLE_DEVICES=0 python get_activations.py llama2_chat_13B tqa_mc2 --device 0
# CUDA_VISIBLE_DEVICES=0 python get_activations.py llama2_chat_13B tqa_gen_end_q --device 0

# python get_activations.py llama2_chat_70B tqa_mc2 --device 0
# python get_activations.py llama2_chat_70B tqa_gen_end_q --device 0