WITH cohort AS (
  SELECT d.subject_id, d.hadm_id, d.stay_id,
         d.gender, d.admission_age AS anchor_age, d.race,
         a.admission_type, a.admission_location, a.insurance, a.marital_status,
         i.first_careunit,
         d.icu_intime AS intime, d.icu_outtime AS outtime,
         TIMESTAMP_ADD(d.icu_intime, INTERVAL 24 HOUR) AS prediction_time,
         EXTRACT(HOUR FROM d.icu_intime) AS icu_admission_hour,
         IF(EXTRACT(DAYOFWEEK FROM d.icu_intime) IN (1, 7), 1, 0) AS weekend_admission,
         IF(a.admission_type LIKE '%EMER%' OR a.admission_type LIKE '%URGENT%', 1, 0) AS emergency_admission,
         IF(a.admission_location LIKE '%TRANSFER%', 1, 0) AS transfer_admission,
         d.hospital_expire_flag
  FROM `physionet-data.mimiciv_3_1_derived.icustay_detail` d
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING (hadm_id)
  JOIN `physionet-data.mimiciv_3_1_icu.icustays` i USING (stay_id)
  WHERE d.first_icu_stay AND d.admission_age >= 18
),
-- first value of each lab within [-6h, +24h] of ICU admission
first_chem AS (
  SELECT c.stay_id,
         ARRAY_AGG(l.creatinine  IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS creatinine_first,
         ARRAY_AGG(l.bun         IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS bun_first,
         ARRAY_AGG(l.sodium      IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS sodium_first,
         ARRAY_AGG(l.potassium   IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS potassium_first,
         ARRAY_AGG(l.glucose     IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS glucose_first,
         ARRAY_AGG(l.albumin     IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS albumin_first,
         ARRAY_AGG(l.bicarbonate IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS bicarbonate_first
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.chemistry` l
    ON l.hadm_id = c.hadm_id
   AND l.charttime BETWEEN TIMESTAMP_SUB(c.intime, INTERVAL 6 HOUR) AND c.prediction_time
  GROUP BY c.stay_id
),
first_cbc AS (
  SELECT c.stay_id,
         ARRAY_AGG(l.wbc        IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS wbc_first,
         ARRAY_AGG(l.platelet   IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS platelets_first,
         ARRAY_AGG(l.hemoglobin IGNORE NULLS ORDER BY l.charttime LIMIT 1)[SAFE_OFFSET(0)] AS hemoglobin_first
  FROM cohort c
  JOIN `physionet-data.mimiciv_3_1_derived.complete_blood_count` l
    ON l.hadm_id = c.hadm_id
   AND l.charttime BETWEEN TIMESTAMP_SUB(c.intime, INTERVAL 6 HOUR) AND c.prediction_time
  GROUP BY c.stay_id
),
first_other AS (
  SELECT c.stay_id,
         ARRAY_AGG(b.lactate IGNORE NULLS ORDER BY b.charttime LIMIT 1)[SAFE_OFFSET(0)] AS lactate_first,
         ARRAY_AGG(e.bilirubin_total IGNORE NULLS ORDER BY e.charttime LIMIT 1)[SAFE_OFFSET(0)] AS bilirubin_total_first
  FROM cohort c
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.bg` b
    ON b.hadm_id = c.hadm_id
   AND b.charttime BETWEEN TIMESTAMP_SUB(c.intime, INTERVAL 6 HOUR) AND c.prediction_time
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.enzyme` e
    ON e.hadm_id = c.hadm_id
   AND e.charttime BETWEEN TIMESTAMP_SUB(c.intime, INTERVAL 6 HOUR) AND c.prediction_time
  GROUP BY c.stay_id
)
SELECT c.*,
       v.heart_rate_min, v.heart_rate_max, v.heart_rate_mean,
       v.resp_rate_min AS respiratory_rate_min, v.resp_rate_max AS respiratory_rate_max, v.resp_rate_mean AS respiratory_rate_mean,
       v.spo2_min, v.spo2_max, v.spo2_mean,
       v.sbp_min, v.sbp_max, v.sbp_mean,
       v.mbp_min AS map_min, v.mbp_max AS map_max, v.mbp_mean AS map_mean,
       v.temperature_min AS temperature_c_min, v.temperature_max AS temperature_c_max, v.temperature_mean AS temperature_c_mean,
       fc.creatinine_first, l.creatinine_min, l.creatinine_max,
       fc.bun_first, l.bun_min, l.bun_max,
       fc.sodium_first, l.sodium_min, l.sodium_max,
       fc.potassium_first, l.potassium_min, l.potassium_max,
       fc.glucose_first, l.glucose_min, l.glucose_max,
       fb.wbc_first, l.wbc_min, l.wbc_max,
       fb.platelets_first, l.platelets_min, l.platelets_max,
       fo.lactate_first, bg.lactate_min, bg.lactate_max,
       fo.bilirubin_total_first, l.bilirubin_total_min, l.bilirubin_total_max,
       fc.albumin_first, l.albumin_min, l.albumin_max,
       fc.bicarbonate_first, l.bicarbonate_min, l.bicarbonate_max,
       fb.hemoglobin_first, l.hemoglobin_min, l.hemoglobin_max
FROM cohort c
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_vitalsign` v USING (stay_id)
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_lab` l USING (stay_id)
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_bg` bg USING (stay_id)
LEFT JOIN first_chem fc USING (stay_id)
LEFT JOIN first_cbc fb USING (stay_id)
LEFT JOIN first_other fo USING (stay_id)