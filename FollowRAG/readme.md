
### inference
You first need to perform inference on followRAG, and the pseudocode is as follows:
```python
followRAG_full=load_json('followRAG/followRAG_full.json')
data_inferenced=[]
for dp in followRAG_full:
    response=llm.inference(dp['prompt'])
    dp['response']=response
    data_inferenced.append(dp)
save_jsonl(data_inferenced,'results/finish_inference/data_inferenced.jsonl')
```
### eval
After completing the inference, run the evaluation script:
```bash
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_API_BASE=https://api.openai.com/v1
python eval/main_eval.py \
    --input_file_path results/finish_inference/data_inferenced.jsonl \
    --output_file_path results/finish_eval/data_evaled.jsonl \
    --rag_eval_type mini \
    --result_log_file_path results/logs/results_log.jsonl
```
Our evaluation of instruction-following part largely draws on the [IFEval code repository](https://github.com/google-research/google-research/tree/master/instruction_following_eval). We appreciate their excellent work!
