#!/bin/bash

set -e
psql -U sample -d postgres -c "create database prefect_db"