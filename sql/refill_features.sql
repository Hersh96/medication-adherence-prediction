-- Refill-behaviour features for each (member, index_day) from the 180-day lookback window.
-- Inputs: fills(member_id, fill_day, days_supply), snapshots(member_id, index_day)
WITH windowed AS (
    SELECT
        s.member_id,
        s.index_day,
        f.fill_day,
        f.days_supply,
        LEAD(f.fill_day) OVER (PARTITION BY s.member_id, s.index_day ORDER BY f.fill_day) AS next_fill_day
    FROM snapshots s
    JOIN fills f
      ON f.member_id = s.member_id
     AND f.fill_day >= s.index_day - 180
     AND f.fill_day <  s.index_day
),
gaps AS (
    SELECT
        *,
        -- Positive = days without medication before the next fill (only when a next fill exists
        -- inside the window; the open-ended last fill is handled by days_until_runout).
        next_fill_day - (fill_day + days_supply) AS gap_days
    FROM windowed
)
SELECT
    member_id,
    index_day,
    COUNT(*)                                                   AS n_fills_180,
    AVG(CASE WHEN days_supply >= 90 THEN 1.0 ELSE 0.0 END)     AS share_90_day_fills,
    COALESCE(MAX(gap_days), 0)                                 AS max_gap_days,
    COALESCE(AVG(GREATEST(gap_days, 0)), 0)                    AS mean_gap_days,
    COALESCE(SUM(CASE WHEN gap_days > 7 THEN 1 ELSE 0 END), 0) AS n_late_fills,
    COALESCE(SUM(CASE WHEN gap_days > 7 AND fill_day >= index_day - 90 THEN 1 ELSE 0 END), 0)
                                                               AS n_late_fills_90,
    MIN(index_day - fill_day)                                  AS days_since_last_fill,
    MAX(fill_day + days_supply) - index_day                    AS days_until_runout,
    MIN(fill_day)                                              AS first_fill_in_window
FROM gaps
GROUP BY member_id, index_day
