-- ============================================================
-- 03_lab_features.sql
--
-- Laboratory summaries during first 24 hours
-- ============================================================

SELECT

    c.stay_id,

    MIN(le.valuenum) AS creatinine_min,

    MAX(le.valuenum) AS creatinine_max,

    AVG(le.valuenum) AS creatinine_mean

FROM first_icu_stay c

JOIN labevents le

ON c.hadm_id = le.hadm_id

WHERE

    le.itemid = 50912       -- Creatinine

    AND le.charttime >= c.intime

    AND le.charttime <= c.intime + INTERVAL '24 hours'

GROUP BY

    c.stay_id;