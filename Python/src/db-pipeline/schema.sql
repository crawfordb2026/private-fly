-- Python/src/db-pipeline/schema.sql
-- Clean, fresh-start schema for a single active experiment workflow.
-- This script intentionally resets the public schema.

DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;

-- Optional TimescaleDB extension (graceful fallback to regular PostgreSQL tables)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'TimescaleDB extension not available. Using regular PostgreSQL tables.';
END $$;

CREATE TABLE experiments (
    experiment_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    lights_on_hour INT NOT NULL DEFAULT 9,
    lights_off_hour INT NOT NULL DEFAULT 21,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE flies (
    fly_id VARCHAR(50) NOT NULL,
    experiment_id INT NOT NULL REFERENCES experiments(experiment_id),
    monitor VARCHAR(50) NOT NULL,
    channel INT NOT NULL,
    genotype VARCHAR(50) NOT NULL,
    sex VARCHAR(20) NOT NULL,
    treatment VARCHAR(100) NOT NULL,
    death_datetime TIMESTAMP NULL,
    death_exp_day INT NULL,
    PRIMARY KEY (fly_id, experiment_id),
    UNIQUE (experiment_id, monitor, channel)
);

CREATE INDEX idx_flies_experiment ON flies(experiment_id);

CREATE TABLE readings (
    measurement_id BIGSERIAL,
    experiment_id INT NOT NULL REFERENCES experiments(experiment_id),
    fly_id VARCHAR(50) NOT NULL,
    datetime TIMESTAMP NOT NULL,
    reading_type VARCHAR(10) NOT NULL CHECK (reading_type IN ('MT', 'CT', 'Pn')),
    value INT NOT NULL,
    monitor VARCHAR(50) NOT NULL,
    PRIMARY KEY (measurement_id, datetime),
    FOREIGN KEY (fly_id, experiment_id) REFERENCES flies(fly_id, experiment_id)
);

DO $$
BEGIN
    PERFORM create_hypertable(
        'readings',
        'datetime',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
    );
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'TimescaleDB not available. Using regular PostgreSQL table for readings.';
END $$;

CREATE INDEX idx_readings_fly_datetime ON readings(fly_id, datetime);
CREATE INDEX idx_readings_experiment ON readings(experiment_id);
CREATE INDEX idx_readings_experiment_fly ON readings(experiment_id, fly_id);

CREATE TABLE health_reports (
    health_report_id SERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES experiments(experiment_id),
    fly_id VARCHAR(50) NOT NULL,
    report_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('Alive', 'Unhealthy', 'Dead', 'QC_Fail')),
    total_activity INT,
    longest_zero_hours DECIMAL(5,2),
    rel_activity DECIMAL(5,3),
    has_startle_response BOOLEAN,
    missing_fraction DECIMAL(5,3),
    UNIQUE (fly_id, experiment_id, report_date),
    FOREIGN KEY (fly_id, experiment_id) REFERENCES flies(fly_id, experiment_id)
);

CREATE INDEX idx_health_fly ON health_reports(fly_id, experiment_id);

CREATE TABLE features (
    feature_id SERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES experiments(experiment_id),
    fly_id VARCHAR(50) NOT NULL,
    mesor_mean DECIMAL(10,3),
    mesor_sd DECIMAL(10,3),
    amplitude_mean DECIMAL(10,3),
    amplitude_sd DECIMAL(10,3),
    phase_mean DECIMAL(10,3),
    phase_sd DECIMAL(10,3),
    rhythmic_days INT,
    periodogram_period_mean DECIMAL(10,3),
    periodogram_period_sd DECIMAL(10,3),
    periodogram_power_mean DECIMAL(10,6),
    total_sleep_mean DECIMAL(10,2),
    day_sleep_mean DECIMAL(10,2),
    night_sleep_mean DECIMAL(10,2),
    total_bouts_mean DECIMAL(10,2),
    day_bouts_mean DECIMAL(10,2),
    night_bouts_mean DECIMAL(10,2),
    mean_bout_mean DECIMAL(10,2),
    max_bout_mean DECIMAL(10,2),
    mean_day_bout_mean DECIMAL(10,2),
    max_day_bout_mean DECIMAL(10,2),
    mean_night_bout_mean DECIMAL(10,2),
    max_night_bout_mean DECIMAL(10,2),
    frag_bouts_per_hour_mean DECIMAL(10,4),
    frag_bouts_per_min_sleep_mean DECIMAL(10,4),
    mean_wake_bout_mean DECIMAL(10,2),
    p_wake_mean DECIMAL(10,4),
    p_doze_mean DECIMAL(10,4),
    sleep_latency_mean DECIMAL(10,2),
    waso_mean DECIMAL(10,2),
    UNIQUE (fly_id, experiment_id),
    FOREIGN KEY (fly_id, experiment_id) REFERENCES flies(fly_id, experiment_id)
);

CREATE TABLE features_sliding_window (
    experiment_id INT NOT NULL REFERENCES experiments(experiment_id),
    fly_id VARCHAR(50) NOT NULL,
    window_end_date DATE NOT NULL,
    exp_day INT,
    total_activity DECIMAL(12,2),
    activity_mean DECIMAL(10,4),
    activity_var DECIMAL(12,4),
    longest_zero_hours DECIMAL(6,2),
    total_sleep_min DECIMAL(10,2),
    total_bouts INT,
    mean_bout_min DECIMAL(10,2),
    max_bout_min DECIMAL(10,2),
    frag_bouts_per_hour DECIMAL(10,4),
    amplitude_24h DECIMAL(10,4),
    periodogram_period_24h DECIMAL(10,3),
    periodogram_power_24h DECIMAL(10,6),
    status VARCHAR(20) NOT NULL,
    status_raw VARCHAR(20),
    days_until_death INT NULL,
    PRIMARY KEY (fly_id, experiment_id, window_end_date),
    FOREIGN KEY (fly_id, experiment_id) REFERENCES flies(fly_id, experiment_id)
);

CREATE INDEX idx_features_sliding_window_experiment ON features_sliding_window(experiment_id);

CREATE TABLE features_z (
    feature_id SERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES experiments(experiment_id),
    fly_id VARCHAR(50) NOT NULL,
    mesor_mean_z DECIMAL(10,6),
    mesor_sd_z DECIMAL(10,6),
    amplitude_mean_z DECIMAL(10,6),
    amplitude_sd_z DECIMAL(10,6),
    phase_mean_z DECIMAL(10,6),
    phase_sd_z DECIMAL(10,6),
    periodogram_period_mean_z DECIMAL(10,6),
    periodogram_period_sd_z DECIMAL(10,6),
    periodogram_power_mean_z DECIMAL(10,6),
    total_sleep_mean_z DECIMAL(10,6),
    day_sleep_mean_z DECIMAL(10,6),
    night_sleep_mean_z DECIMAL(10,6),
    total_bouts_mean_z DECIMAL(10,6),
    day_bouts_mean_z DECIMAL(10,6),
    night_bouts_mean_z DECIMAL(10,6),
    mean_bout_mean_z DECIMAL(10,6),
    max_bout_mean_z DECIMAL(10,6),
    mean_day_bout_mean_z DECIMAL(10,6),
    max_day_bout_mean_z DECIMAL(10,6),
    mean_night_bout_mean_z DECIMAL(10,6),
    max_night_bout_mean_z DECIMAL(10,6),
    frag_bouts_per_hour_mean_z DECIMAL(10,6),
    frag_bouts_per_min_sleep_mean_z DECIMAL(10,6),
    mean_wake_bout_mean_z DECIMAL(10,6),
    p_wake_mean_z DECIMAL(10,6),
    p_doze_mean_z DECIMAL(10,6),
    sleep_latency_mean_z DECIMAL(10,6),
    waso_mean_z DECIMAL(10,6),
    UNIQUE (fly_id, experiment_id),
    FOREIGN KEY (fly_id, experiment_id) REFERENCES flies(fly_id, experiment_id)
);

CREATE TABLE hmm_states (
    experiment_id INT NOT NULL REFERENCES experiments(experiment_id),
    fly_id VARCHAR(50) NOT NULL,
    bin_index INT NOT NULL,
    bin_minutes INT NOT NULL,
    state INT NOT NULL CHECK (state BETWEEN 0 AND 3),
    state_label VARCHAR(20) NOT NULL CHECK (
        state_label IN ('Healthy','Declining','Critical','Dead','Sleep','Warmup','Low_Data')
    ),
    PRIMARY KEY (experiment_id, fly_id, bin_index),
    FOREIGN KEY (fly_id, experiment_id) REFERENCES flies(fly_id, experiment_id)
);

CREATE INDEX idx_hmm_states_fly ON hmm_states(fly_id, experiment_id);