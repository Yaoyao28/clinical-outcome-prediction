-- ============================================================
-- 02_vital_features.sql
--
-- Heart rate summary during first 24 hours
-- ============================================================

SELECT

    c.stay_id,

    MIN(ce.valuenum) AS heart_rate_min,

    MAX(ce.valuenum) AS heart_rate_max,

    AVG(ce.valuenum) AS heart_rate_mean

FROM first_icu_stay c

JOIN chartevents ce

ON c.stay_id = ce.stay_id

WHERE

    ce.itemid = 220045      -- Heart Rate

    AND ce.charttime >= c.intime

    AND ce.charttime <= c.intime + INTERVAL '24 hours'

GROUP BY

    c.stay_id;