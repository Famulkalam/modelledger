.PHONY: install keys train seed attack-naive attack-recompute attack-outcome charts report test run docs docker-up docker-down demo-flow clean

install:
	pip install -r requirements.txt

keys:            ## print a new Ed25519 key pair for .env
	python scripts/generate_keys.py

train:           ## train the XGBoost model (downloads German Credit from OpenML once)
	python scripts/train_model.py

seed:            ## log 300 held-out applicants over 30 days, with outcomes and anchors
	python scripts/seed_demo_data.py

attack-naive:     ## edit one decision via SQL: caught by the hash chain
	python scripts/seed_demo_data.py --attack naive

attack-recompute: ## edit one decision and recompute every hash: caught by signed anchors
	python scripts/seed_demo_data.py --attack recompute

attack-outcome:   ## rewrite a repayment outcome: caught by the outcome chain
	python scripts/seed_demo_data.py --attack outcome

charts:
	python scripts/generate_charts.py

report:          ## write a PDF evidence pack to reports/
	python scripts/make_report.py

test:
	python -m pytest -q

run:
	uvicorn modelledger.api.server:app --reload --port 8000

docs:
	python scripts/export_openapi.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

demo-flow:       ## needs `make train` and a .env with keys first
	docker compose up --build -d
	sleep 8
	curl -s -X POST http://localhost:8001/predict -H 'Content-Type: application/json' \
		-d '{"checking_status":"<0","duration":36,"credit_amount":9000,"age":23,"personal_status":"female div/dep/mar"}' | python3 -m json.tool
	sleep 2
	curl -s -H "X-API-Key: $$(grep ^MODELLEDGER_API_KEY .env | cut -d= -f2)" http://localhost:8000/verify | python3 -m json.tool

clean:
	rm -f data/modelledger.db* reports/*.pdf anchors/*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
