.DEFAULT_GOAL := help

#help:				@ list available goals
.PHONY: help
help:
	@grep -E '[a-zA-Z\.\-]+:.*?@ .*$$' $(MAKEFILE_LIST)| sort | tr -d '#'  | awk 'BEGIN {FS = ":.*?@ "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

#setup:				@ install dependencies configured in pyproject.toml
.PHONY: setup
setup:
	@echo " install dependencies"
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	poetry install --no-interaction

#test:				@ run test with docker
.PHONY: test
test:
	@echo " running tests"
	docker compose down
	docker compose up --build test
	docker compose down

#run:				@ run application in docker
.PHONY: run
run:
	@echo " running service"
	docker compose down
	docker compose up --build app


#train:			   @ train a gradient boosting
.PHONY: train
train:
	@echo " training model"
	docker compose down
	docker compose up --build train

#get_embedings:			   @ get_embedings
.PHONY: get_embedings
get_embedings:
	@echo " get embedings "
	wget -P storage/embedings https://nlp.stanford.edu/data/glove.6B.zip
	unzip storage/embedings/glove.6B.zip -d storage/embedings/
	rm -f storage/embedings/glove.6B.zip
