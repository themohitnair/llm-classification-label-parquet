# llm-classification-label-parquet

A workflow to label text examples (social media posts) from a postgres database using an LLM and store it in a parquet file.

## Example environment `.env`

```.env
MODEL_BASE_URL="..."
MODEL_API_KEY="..."
PSQL_DB_USERNAME="..."
PSQL_DB_PWD="..."
PSQL_DB_HOSTNAME="..."
PSQL_DB="..."
PSQL_DB_PORT=...
PSQL_TABLE=...
MODEL="..."
```
