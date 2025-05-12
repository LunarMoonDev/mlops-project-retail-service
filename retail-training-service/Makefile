# for pretty logs
GREEN   := \033[0;32m
BLUE    := \033[0;34m
RESET   := \033[0m

# function in makefile
echo_message = \
        @echo -e "$(BLUE)********************" \
         " $(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')] $1 " \
         "$(BLUE)********************$(RESET) \n"

# targets		 
quality_checks:
		$(call echo_message,[QUALITY_CHECK] Starting quality check...)
		isort --diff .
		black --diff .
		pylint --recursive=y .

setup_local:
		$(call echo_message,[SETUP_LOCAL] Setting up local...)
		bash scripts/create_buckets.sh
		bash scripts/push_data.sh

run_local:
		$(call echo_message,[RUN_LOCAL] Executing Prefect Locally...)
		PREFECT_API_URL=http://localhost:4200/api AWS_PROFILE=minio python -m main