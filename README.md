# IntelliFoods Backend
## How to setup the python env

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run the code

```bash
uvicorn main:app --reload
```

## How to run the code in Docker

```bash
docker build -t recipe-api .

docker run --rm -p 8000:8000 \
    --entrypoint python \
    --env-file .env \
    recipe-api main.py
```