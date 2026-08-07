-- ============================================================
-- 01_first_icu_stay.sql
--
-- Keep the first ICU stay for every hospital admission.
-- ============================================================

WITH ranked_stays AS (


    SELECT

        subject_id,
        hadm_id,
        stay_id,
        first_careunit,
        last_careunit,
        intime,
        outtime,

        ROW_NUMBER() OVER (

            PARTITION BY hadm_id
            ORDER BY intime

        ) AS stay_rank

    FROM icustays

)

SELECT

    subject_id,
    hadm_id,
    stay_id,
   first_careunit,
    last_careunit,
    intime,
    outtime

FROM ranked_stays

WHERE stay_rank = 1;