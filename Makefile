LOGS_DIR := logs
SCRATCH := /scratch/$${USER}
REPO_DIR := $(SCRATCH)/slowfast
MAIL_ADDRESS := $${USER}@nyu.edu
DURATION := 12:00:00
WANDB_CACHE_DIR := $(SCRATCH)/.cache/wandb
WANDB_DATA_DIR := $(SCRATCH)/.cache/wandb/data

.PHONY: bash
bash:
	@echo "Running interactive bash session"
	@srun --job-name "interactive bash" \
		--cpus-per-task 4 \
		--mem 64G \
		--time 4:00:00 \
		--pty bash


.PHONY: bash-gpu
bash-gpu:
	@echo "Running interactive bash session"
	@srun --job-name "bash-gpu" \
		--cpus-per-task 8 \
		--mem 16G \
		--gres gpu:1 \
		--time 4:00:00 \
		--pty bash

.PHONY: job-test
job-test:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME="sf-test-cm"; \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 32G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make test"


.PHONY: test
test:
	@echo "Running the main script"
	@./singrw <<< "python tools/run_net.py"

.PHONY: train
train:
	@echo "Running the main script"
	@./singrw <<< "python tools/run_net.py"
