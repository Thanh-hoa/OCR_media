run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

dev:
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

install:
	pip install -r requirements.txt

health:
	curl -s http://127.0.0.1:8000/health

docker-build:
	docker build -t project-ocr-api .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env project-ocr-api

.PHONY: run prod dev install health docker-build docker-run
