-- ============================================================
-- 04_modeling_cohort.sql
--
-- Final modeling dataset
-- ============================================================

SELECT

    d.*,

    v.heart_rate_min,
    v.heart_rate_max,
    v.heart_rate_mean,

    l.creatinine_min,
    l.creatinine_max,
    l.creatinine_mean

FROM demographic_features d

LEFT JOIN vital_features v

ON d.stay_id = v.stay_id

LEFT JOIN lab_features l

ON d.stay_id = l.stay_id;