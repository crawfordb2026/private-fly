## ============================================================
## HMM: Automated Fly Health QC Screen
## Purpose:   Flag Dead flies for exclusion before downstream
##            DAM analysis
## States:    Sleep     — pre-labeled (MT_sum == 0)
##            Warmup    — pre-labeled (< 12h active bins elapsed)
##            Low_Data  — pre-labeled (rolling SD near zero)
##            Healthy   \
##            Declining   > inferred by HMM on qualifying active bins
##            Critical  /
##            Dead      — pre-labeled (144 consec bins MT==0 & Pn==0)
## Emissions: Rolling z-score of MT_sum and Pn_var
##            (residual / rolling SD, 24h trailing window,
##             active bins only, per fly)
## Death:     12 consecutive hours zero MT AND zero Pn_var
## Genotype:  Separate HMM per genotype, pooled across flies
## ============================================================

## 1. LIBRARIES ---------------------------------------------
library(DBI)
library(RPostgres)
library(dplyr)
library(lubridate)
library(tidyr)
library(zoo)        # for rollapply
library(depmixS4)

select <- dplyr::select


## 2. LOAD DATA ---------------------------------------------
df <- readRDS("C:/Users/Bedont Lab/Desktop/Flyght Patterns R/df_processed_MT_Pn.rds")


## 3. BIN TO 5-MINUTE WINDOWS (raw) ------------------------
df_wide <- df %>%
  select(fly_id, datetime, reading_type, value, genotype, sex, treatment) %>%
  pivot_wider(names_from = reading_type, values_from = value) %>%
  filter(!is.na(MT), !is.na(Pn)) %>%
  arrange(fly_id, datetime) %>%
  mutate(bin_start = floor_date(datetime, "5 minutes")) %>%
  group_by(fly_id, bin_start, genotype, sex, treatment) %>%
  summarise(
    n_obs  = n(),
    MT_sum = sum(MT, na.rm = TRUE),
    Pn_var = var(Pn, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_obs == 5) %>%
  mutate(Pn_var = ifelse(is.na(Pn_var), 0, Pn_var)) %>%
  select(-n_obs)


## 4. DETECT TIME OF DEATH (raw values) --------------------
# Death = first onset of 144 consecutive 5-min bins where
# MT_sum == 0 AND Pn_var == 0  (12 hours at 5-min resolution)
DEATH_BINS    <- 144  # 12 hours
MIN_BOUT_BINS <- 6    # 30 min — shorter active bouts relabeled Sleep

# Rolling window parameters
ROLL_BINS     <- 288  # 24 hours at 5-min resolution
WARMUP_ACTIVE <- 144  # 12 hours of active bins required before HMM fitting
SD_FLOOR_MULT <- 0.01 # rolling SD below 1% of fly's own active SD = Low_Data

tod_table <- df_wide %>%
  arrange(fly_id, bin_start) %>%
  group_by(fly_id) %>%
  summarise(
    time_of_death = {
      dead_vec <- (MT_sum == 0 & Pn_var == 0)
      n        <- length(dead_vec)
      tod      <- as.POSIXct(NA)
      if (n >= DEATH_BINS) {
        for (i in 1:(n - DEATH_BINS + 1)) {
          if (all(dead_vec[i:(i + DEATH_BINS - 1)])) {
            tod <- bin_start[i]
            break
          }
        }
      }
      tod
    },
    .groups = "drop"
  )

df_wide <- df_wide %>%
  left_join(tod_table, by = "fly_id") %>%
  mutate(is_post_death = !is.na(time_of_death) & bin_start >= time_of_death)

cat("\nFlies with detected time of death:",
    sum(!is.na(tod_table$time_of_death)), "/", nrow(tod_table), "\n")


## 5. PRE-LABEL SLEEP, DEAD, AND SHORT BOUTS ---------------
df_wide <- df_wide %>%
  mutate(
    raw_state = case_when(
      is_post_death ~ "Dead",
      MT_sum == 0   ~ "Sleep",
      TRUE          ~ "Active"
    )
  )

# Relabel active bouts shorter than MIN_BOUT_BINS as Sleep
df_wide <- df_wide %>%
  arrange(fly_id, bin_start) %>%
  group_by(fly_id) %>%
  mutate(
    rle_id   = cumsum(raw_state != lag(raw_state,
                                       default = first(raw_state))),
    bout_len = ave(raw_state, rle_id, FUN = length),
    raw_state = ifelse(
      raw_state == "Active" & as.integer(bout_len) < MIN_BOUT_BINS,
      "Sleep",
      raw_state
    )
  ) %>%
  ungroup() %>%
  select(-rle_id, -bout_len)

cat("\nBin state breakdown after sleep pre-labeling:\n")
df_wide %>%
  count(raw_state) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  print()


## 6. ROLLING BASELINE Z-SCORES (per fly, active bins only) -
# For each active bin compute:
#   MT_roll_z = (MT_sum - rolling_mean) / rolling_sd
#   Pn_roll_z = (Pn_var - rolling_mean) / rolling_sd
# Rolling stats use the trailing 24h of ACTIVE bins only.
# Bins in the warmup period (< WARMUP_ACTIVE active bins elapsed)
# are labeled Warmup and excluded from HMM fitting.
# Bins where rolling SD < SD_FLOOR_MULT * fly_sd are labeled Low_Data.

df_wide <- df_wide %>%
  arrange(fly_id, bin_start) %>%
  group_by(fly_id) %>%
  mutate(
    # Per-fly SD over all active bins (used for Low_Data floor)
    fly_mt_sd = sd(MT_sum[raw_state == "Active"], na.rm = TRUE),
    fly_pn_sd = sd(Pn_var[raw_state == "Active"], na.rm = TRUE),
    
    # Cumulative count of active bins seen so far per fly
    active_cumcount = cumsum(raw_state == "Active"),
    
    # Rolling mean and SD over trailing ROLL_BINS active bins
    # NA for non-active bins — filled via na.locf below
    MT_roll_mean = ifelse(
      raw_state == "Active",
      rollapply(
        ifelse(raw_state == "Active", MT_sum, NA_real_),
        width   = ROLL_BINS,
        FUN     = function(x) mean(x[!is.na(x)], na.rm = TRUE),
        fill    = NA,
        align   = "right",
        partial = TRUE
      ),
      NA_real_
    ),
    MT_roll_sd = ifelse(
      raw_state == "Active",
      rollapply(
        ifelse(raw_state == "Active", MT_sum, NA_real_),
        width   = ROLL_BINS,
        FUN     = function(x) sd(x[!is.na(x)], na.rm = TRUE),
        fill    = NA,
        align   = "right",
        partial = TRUE
      ),
      NA_real_
    ),
    Pn_roll_mean = ifelse(
      raw_state == "Active",
      rollapply(
        ifelse(raw_state == "Active", Pn_var, NA_real_),
        width   = ROLL_BINS,
        FUN     = function(x) mean(x[!is.na(x)], na.rm = TRUE),
        fill    = NA,
        align   = "right",
        partial = TRUE
      ),
      NA_real_
    ),
    Pn_roll_sd = ifelse(
      raw_state == "Active",
      rollapply(
        ifelse(raw_state == "Active", Pn_var, NA_real_),
        width   = ROLL_BINS,
        FUN     = function(x) sd(x[!is.na(x)], na.rm = TRUE),
        fill    = NA,
        align   = "right",
        partial = TRUE
      ),
      NA_real_
    )
  ) %>%
  mutate(
    # SD floor: 1% of fly's own active SD to avoid near-zero division
    mt_sd_floor = SD_FLOOR_MULT * fly_mt_sd,
    pn_sd_floor = SD_FLOOR_MULT * fly_pn_sd,
    
    # Rolling z-scores
    MT_roll_z = (MT_sum - MT_roll_mean) /
      pmax(MT_roll_sd, mt_sd_floor, na.rm = TRUE),
    Pn_roll_z = (Pn_var - Pn_roll_mean) /
      pmax(Pn_roll_sd, pn_sd_floor, na.rm = TRUE),
    
    # Final state label for each bin
    hmm_state = case_when(
      raw_state == "Dead"                              ~ "Dead",
      raw_state == "Sleep"                             ~ "Sleep",
      raw_state == "Active" &
        active_cumcount < WARMUP_ACTIVE                ~ "Warmup",
      raw_state == "Active" &
        (MT_roll_sd < mt_sd_floor |
           Pn_roll_sd < pn_sd_floor)                     ~ "Low_Data",
      raw_state == "Active"                            ~ "HMM_candidate",
      TRUE                                             ~ "Sleep"
    )
  ) %>%
  ungroup() %>%
  select(-fly_mt_sd, -fly_pn_sd, -mt_sd_floor, -pn_sd_floor,
         -MT_roll_mean, -MT_roll_sd, -Pn_roll_mean, -Pn_roll_sd)

cat("\nBin hmm_state breakdown:\n")
df_wide %>%
  count(hmm_state) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  arrange(desc(n)) %>%
  print()


## 7. HELPERS -----------------------------------------------

get_start_vals <- function(data_df, spread_factor = 1.0) {
  lo   <- max(0.05, 0.15 - 0.10 * (spread_factor - 1))
  hi   <- min(0.95, 0.85 + 0.10 * (spread_factor - 1))
  mt_q <- quantile(data_df$MT_roll_z, probs = c(lo, 0.50, hi), na.rm = TRUE)
  pn_q <- quantile(data_df$Pn_roll_z, probs = c(lo, 0.50, hi), na.rm = TRUE)
  list(mt_means = as.numeric(mt_q), pn_means = as.numeric(pn_q))
}

inject_start_vals <- function(mod, sv, n_states) {
  pars        <- getpars(mod)
  emit_offset <- n_states + n_states^2
  for (s in 1:n_states) {
    pars[emit_offset + (s - 1) * 4 + 1] <- sv$mt_means[s]
    pars[emit_offset + (s - 1) * 4 + 3] <- sv$pn_means[s]
  }
  setpars(mod, pars)
}

fit_with_retry <- function(mod_template, geno_df, n_states, max_tries = 4) {
  for (attempt in 1:max_tries) {
    spread <- 1.0 + (attempt - 1) * 0.5
    sv     <- get_start_vals(geno_df, spread_factor = spread)
    mod    <- inject_start_vals(mod_template, sv, n_states)
    result <- tryCatch(
      fit(mod, verbose = FALSE,
          emcontrol = em.control(maxit = 500, tol = 1e-8)),
      error = function(e) NULL
    )
    if (!is.null(result)) {
      cat("  Converged on attempt", attempt,
          "(spread_factor =", spread, ")\n")
      return(result)
    }
    cat("  Attempt", attempt, "failed (spread_factor =", spread,
        ") — retrying...\n")
  }
  NULL
}

label_states <- function(viterbi_states, active_df, n_states, state_labels) {
  mt_means <- sapply(1:n_states, function(s)
    mean(active_df$MT_roll_z[viterbi_states == s], na.rm = TRUE))
  pn_means <- sapply(1:n_states, function(s)
    mean(active_df$Pn_roll_z[viterbi_states == s], na.rm = TRUE))
  
  state_order <- order(-mt_means, -pn_means)
  label_map   <- setNames(state_labels, state_order)
  
  cat("  State emission means (MT_roll_z / Pn_roll_z):\n")
  for (s in 1:n_states) {
    cat(sprintf("    State %d: MT_z=%.3f  Pn_z=%.3f  -> %s\n",
                s, mt_means[s], pn_means[s],
                label_map[as.character(s)]))
  }
  
  label_map[as.character(viterbi_states)]
}


## 8. FIT HMM PER GENOTYPE (HMM_candidate bins only) -------
n_states         <- 3
state_labels     <- c("Healthy", "Declining", "Critical")
genotypes        <- unique(df_wide$genotype)
all_state_seqs   <- list()
all_trans_mats   <- list()
failed_genotypes <- character(0)

for (geno in genotypes) {
  
  cat("\n--------------------------------------------\n")
  cat("Fitting HMM for genotype:", geno, "\n")
  
  # Only HMM_candidate bins for this genotype
  cand_df <- df_wide %>%
    filter(genotype == geno, hmm_state == "HMM_candidate") %>%
    arrange(fly_id, bin_start)
  
  # Bout structure: consecutive candidate runs within each fly
  cand_df <- cand_df %>%
    group_by(fly_id) %>%
    mutate(
      time_gap = as.numeric(difftime(
        bin_start,
        lag(bin_start, default = first(bin_start)),
        units = "mins")),
      bout_id = cumsum(time_gap > 5)
    ) %>%
    ungroup()
  
  bout_lengths <- cand_df %>%
    group_by(fly_id, bout_id) %>%
    summarise(n = n(), .groups = "drop") %>%
    filter(n >= MIN_BOUT_BINS)
  
  cand_df <- cand_df %>%
    semi_join(bout_lengths, by = c("fly_id", "bout_id")) %>%
    arrange(fly_id, bout_id, bin_start)
  
  ntimes_vec <- bout_lengths %>%
    arrange(fly_id, bout_id) %>%
    pull(n)
  
  if (nrow(cand_df) < 200 || length(ntimes_vec) == 0) {
    message("Skipping genotype ", geno, " — insufficient candidate bins")
    failed_genotypes <- c(failed_genotypes, geno)
    next
  }
  
  cat("  Candidate bins:", nrow(cand_df),
      "| Bouts:", length(ntimes_vec),
      "| Median bout length:", median(ntimes_vec), "bins\n")
  
  mod_template <- depmix(
    list(MT_roll_z ~ 1, Pn_roll_z ~ 1),
    data    = cand_df,
    nstates = n_states,
    family  = list(gaussian(), gaussian()),
    ntimes  = ntimes_vec
  )
  
  fit_mod <- fit_with_retry(mod_template, cand_df, n_states)
  
  if (is.null(fit_mod)) {
    message("HMM failed for genotype ", geno,
            " after all retries — flies will be flagged HMM_FAILED")
    failed_genotypes <- c(failed_genotypes, geno)
    next
  }
  
  viterbi_states <- posterior(fit_mod, type = "viterbi")$state
  named_states   <- label_states(viterbi_states, cand_df, n_states, state_labels)
  
  # HMM-labeled bins
  hmm_seq <- cand_df %>%
    select(fly_id, bin_start, genotype, sex, treatment,
           MT_sum, Pn_var, MT_roll_z, Pn_roll_z) %>%
    mutate(state = named_states)
  
  # Non-HMM bins: Sleep, Warmup, Low_Data, Dead
  non_hmm_seq <- df_wide %>%
    filter(genotype == geno,
           hmm_state %in% c("Sleep", "Warmup", "Low_Data", "Dead")) %>%
    select(fly_id, bin_start, genotype, sex, treatment,
           MT_sum, Pn_var, MT_roll_z, Pn_roll_z) %>%
    mutate(state = df_wide$hmm_state[
      df_wide$genotype == geno &
        df_wide$hmm_state %in% c("Sleep", "Warmup", "Low_Data", "Dead")
    ])
  
  geno_seq <- bind_rows(hmm_seq, non_hmm_seq) %>%
    arrange(fly_id, bin_start)
  
  all_state_seqs[[geno]] <- geno_seq
  
  trans_pars <- matrix(
    getpars(fit_mod)[(n_states + 1):(n_states + n_states^2)],
    nrow = n_states, byrow = TRUE
  )
  rownames(trans_pars) <- state_labels
  colnames(trans_pars) <- state_labels
  all_trans_mats[[geno]] <- trans_pars
  
  cat("  Transition matrix:\n")
  print(round(trans_pars, 4))
  cat("  Log-likelihood:", logLik(fit_mod), "\n")
}


## 9. COMBINE STATE SEQUENCES --------------------------------
all_state_seqs_df <- bind_rows(all_state_seqs)


## 10. BUILD QC FLAG TABLE ----------------------------------
exp_start <- df_wide %>%
  group_by(fly_id) %>%
  summarise(exp_start = min(bin_start), .groups = "drop")

fly_meta <- df %>%
  distinct(fly_id, genotype, sex, treatment)

# Stub rows for flies whose genotype HMM failed
failed_fly_stubs <- fly_meta %>%
  filter(genotype %in% failed_genotypes) %>%
  left_join(tod_table, by = "fly_id") %>%
  left_join(exp_start, by = "fly_id") %>%
  mutate(
    qc_flag             = "HMM_FAILED",
    total_bins          = NA_integer_,
    pct_healthy         = NA_real_,
    pct_declining       = NA_real_,
    pct_critical        = NA_real_,
    pct_sleep           = NA_real_,
    pct_warmup          = NA_real_,
    pct_low_data        = NA_real_,
    pct_dead            = NA_real_,
    first_declining_day = NA_real_,
    first_critical_day  = NA_real_,
    death_exp_day       = as.numeric(
      difftime(time_of_death, exp_start, units = "days"))
  ) %>%
  select(fly_id, genotype, sex, treatment, qc_flag,
         death_exp_day, time_of_death,
         pct_healthy, pct_declining, pct_critical,
         pct_sleep, pct_warmup, pct_low_data, pct_dead,
         first_declining_day, first_critical_day, total_bins)

# Main QC table
qc_table <- all_state_seqs_df %>%
  left_join(exp_start, by = "fly_id") %>%
  group_by(fly_id) %>%
  summarise(
    
    total_bins    = n(),
    pct_healthy   = round(100 * mean(state == "Healthy"),   1),
    pct_declining = round(100 * mean(state == "Declining"), 1),
    pct_critical  = round(100 * mean(state == "Critical"),  1),
    pct_sleep     = round(100 * mean(state == "Sleep"),     1),
    pct_warmup    = round(100 * mean(state == "Warmup"),    1),
    pct_low_data  = round(100 * mean(state == "Low_Data"),  1),
    pct_dead      = round(100 * mean(state == "Dead"),      1),
    
    first_declining_day = {
      idx <- which(state == "Declining")
      if (length(idx) > 0)
        as.numeric(difftime(bin_start[idx[1]], exp_start[1], units = "days"))
      else NA_real_
    },
    
    first_critical_day = {
      idx <- which(state == "Critical")
      if (length(idx) > 0)
        as.numeric(difftime(bin_start[idx[1]], exp_start[1], units = "days"))
      else NA_real_
    },
    
    .groups = "drop"
  ) %>%
  left_join(tod_table, by = "fly_id") %>%
  left_join(exp_start, by = "fly_id") %>%
  mutate(
    death_exp_day = as.numeric(
      difftime(time_of_death, exp_start, units = "days"))
  ) %>%
  select(-exp_start) %>%
  left_join(fly_meta, by = "fly_id") %>%
  mutate(
    qc_flag = case_when(
      pct_dead > 0 ~ "Exclude",
      TRUE         ~ "Keep"
    )
  ) %>%
  select(fly_id, genotype, sex, treatment, qc_flag,
         death_exp_day, time_of_death,
         pct_healthy, pct_declining, pct_critical,
         pct_sleep, pct_warmup, pct_low_data, pct_dead,
         first_declining_day, first_critical_day, total_bins)

# Merge with failed stubs
qc_table <- bind_rows(qc_table, failed_fly_stubs) %>%
  arrange(qc_flag, genotype, sex, fly_id)


## 11. PRINT QC SUMMARY -------------------------------------
cat("\n============================================\n")
cat("QC SCREEN SUMMARY\n")
cat("============================================\n")

qc_table %>%
  group_by(genotype, qc_flag) %>%
  summarise(
    n_flies            = n(),
    median_pct_healthy = median(pct_healthy,   na.rm = TRUE),
    median_pct_sleep   = median(pct_sleep,     na.rm = TRUE),
    median_pct_warmup  = median(pct_warmup,    na.rm = TRUE),
    median_death_day   = median(death_exp_day, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(genotype) %>%
  mutate(pct_of_geno = round(100 * n_flies / sum(n_flies), 1)) %>%
  ungroup() %>%
  select(genotype, qc_flag, n_flies, pct_of_geno,
         median_pct_healthy, median_pct_sleep,
         median_pct_warmup, median_death_day) %>%
  print()

cat("\nTotal flies screened: ", nrow(qc_table), "\n")
cat("Flagged Exclude:      ", sum(qc_table$qc_flag == "Exclude"),    "\n")
cat("Flagged Keep:         ", sum(qc_table$qc_flag == "Keep"),       "\n")
cat("Flagged HMM_FAILED:   ", sum(qc_table$qc_flag == "HMM_FAILED"), "\n")

if (length(failed_genotypes) > 0) {
  cat("\nGenotypes that failed HMM fitting:",
      paste(failed_genotypes, collapse = ", "), "\n")
  cat("These flies require manual review before downstream analysis.\n")
}


## 12. FULL QC TABLE CONSOLE OUTPUT -------------------------
cat("\n============================================\n")
cat("FULL QC FLAG TABLE\n")
cat("============================================\n")
print(qc_table, n = Inf)

cat("\nTo filter your analysis data, join on fly_id:\n")
cat('  df_clean <- df %>%\n')
cat('    left_join(qc_table %>% select(fly_id, qc_flag), by = "fly_id") %>%\n')
cat('    filter(qc_flag == "Keep")\n')