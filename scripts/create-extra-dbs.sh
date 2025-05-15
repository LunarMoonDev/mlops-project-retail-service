#!/bin/bash

set -e
psql -U sample -d postgres -c "create database training_prefect_db"
psql -U sample -d postgres -c "create database batch_prefect_db"