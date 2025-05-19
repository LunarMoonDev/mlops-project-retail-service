#!/bin/bash

set -e
psql -U sample -d postgres -c "create database training_prefect_db"
psql -U sample -d postgres -c "create database batch_prefect_db"
psql -U sample -d postgres -c "create database retail_db"
psql -U sample -d retail_db -c "
    CREATE TABLE IF NOT EXISTS retail_reviews (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        item_purchased TEXT NOT NULL,
        purchase_amount FLOAT NOT NULL,
        date_purchase DATE NOT NULL,
        review_rating FLOAT NOT NULL,
        payment_method TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP)
"