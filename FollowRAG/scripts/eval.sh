# gpt setting
export OPENAI_API_KEY=xxxxxx
export OPENAI_API_BASE=xxxxx


# eval
python eval/main_eval.py \
    --input_file_path results/finish_inference/data_inferenced.jsonl \
    --output_file_path results/finish_eval/data_evaled.jsonl \
    --rag_eval_type mini \
    --result_log_file_path results/logs/results_log.jsonl 
