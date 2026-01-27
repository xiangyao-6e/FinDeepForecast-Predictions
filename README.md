# FinForecast-Predictions
## Setup Environment
### Setup Conda Environment
Install a conda environment with Python=3.10, then install dependencies with 

```bash
pip install -r requirements.txt
```

### Setup API Keys
Create a `.env` file (or add api keys into `env.py`) and setup api keys for the following:

1. OPENAI_KEY
2. GEMINI_API_KEY
3. CLAUDE_API_KEY
4. GROK_API_KEY
5. OPENROUTER_API_KEY
6. SERPER_KEY_ID
7. JINA_API_KEYS

### Setup Sandbox Fusion for Tongyi

Setup the Docker for Sandbox Fusion. If the 8080 port is not available, you can change it but the corresponding environment variable of `SANDBOX_FUSION_ENDPOINTS` in `env.py` has to be changed as well.

First, download the github repo https://github.com/bytedance/SandboxFusion, then run the following:

```bash
docker build -f ./scripts/Dockerfile.base -t code_sandbox:base .
# change the base image in Dockerfile.server
sed -i '1s/.*/FROM code_sandbox:base/' ./scripts/Dockerfile.server
docker build -f ./scripts/Dockerfile.server -t code_sandbox:server .
docker run -d --rm --privileged -p 8080:8080 code_sandbox:server make run-online
```

## API Predictions

The logic for the predictions is to run all APIs in parallel for a maximum of 3 attempts to account for prediction failures. In each cycle, it will first make predictions, then cleanup failed files (files that does not retrieve Answer). If there is no failed files, then it will break off from the cycle. After 3 attempts, the pipeline will create Answer files for all files regardless of remaining failures.

### Sheet Mapping
You can adjust the Sheet Mapping in the `sheet_mapping.json`. It maps the sheet name in the finforecast excel file to the name of the location for each category in the output. Currently the sheet mapping is set as:

```json
{
    "Corporate - Recurrent": "Corporate_Recurrent",
    "Macro - Recurrent": "Macro_Recurrent",
    "Macro - Non-Recurrent": "Macro_Non_Recurrent",
    "Corporate - Non-Recurrent": "Corporate_Non_Recurrent"
}
```

### Prediction Code

Run the following code:

```bash
PYTHONPATH=. python finforecast/run_all_with_extraction.py \
    --forecast-excel-file /path/to/excel \
    --output-dir /path/to/output
```

A log file will be saved in the `output-dir/pipeline.log` that records the generation and cleanup progress as well as remaining failed files.

## Note

1. The log file will only log summaries during task generation, failure breakdowns when extracting answers from markdown table and verification progress when verifying files. For more verbose details, consider logging the terminal as well by adding `> task.log` at the end of the prediction code.
2. Tongyi Deep Research is made up of different APIs and require the Sandbox Fusion setup. Any one of the APIs may encounter issues, which may cause inaccurate results or task failures. Refer to terminal output for more verbose details.