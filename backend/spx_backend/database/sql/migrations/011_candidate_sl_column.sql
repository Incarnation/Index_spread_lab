-- Migration 011: Add hit_sl_before_tp_or_expiry column to trade_candidates
-- for dual TP+SL label tracking in the labeler job.

ALTER TABLE trade_candidates ADD COLUMN IF NOT EXISTS hit_sl_before_tp_or_expiry BOOLEAN NULL;
