python -m rllte.copilot.controller &
python -m rllte.copilot.model_worker --model-path ../LLM/rllte-vicuna-7b &
sleep 20
python -m rllte.copilot.server