# Neural Forecasting Research

**Online calibration of neural volatility forecasts with early, one-sided feedback.**
Akshat Valse · Iowa State University (Statistics, Mathematics, Data Science, Computer Science) · research project, 2026

This page is a public overview. The code, data pipeline, proofs, experiment logs and write-ups live in a
private repository; I am glad to walk through any of it on request ([GitHub profile](https://github.com/AkshatValse)).

## The problem

Risk forecasts for a *cumulative* quantity, such as the realized variance of an asset over the next six or
twenty-four hours, are only scored when the horizon ends. Every online calibration method therefore learns
with a delay of one full horizon, which is exactly when it is slowest to react to a volatility shift.

Cumulative targets have a structure that is usually ignored: their increments are non-negative, so the
running partial sum is monotone. If the partial sum crosses the issued upper bound before the horizon
ends, the miss is **certain** immediately, while coverage is only confirmed at the end. The feedback is
early but one-sided.

## The idea

A model-agnostic calibration layer that sits on top of any quantile forecaster (a small temporal
convolutional network, a HAR-type regression, or an EWMA model) and

- credits certain early misses to an online integrator as soon as they are confirmed,
- keeps explicit accounts of forecasts that are still pending, so that the eventual miss count is known to
  lie in an interval rather than treated as resolved,
- applies a bounded correction only when the accounting justifies it.

The research question is whether this recovers faster after volatility jumps than delayed-feedback
adaptive conformal inference and the strongest multi-step online conformal baseline, *without*
systematically widening the bounds the rest of the time.

## What has been built

- **Theory.** A long-run coverage guarantee for the calibrator under early crediting, a bounded
  correction and missing data, with tightened constants and a sharpness construction; the proof was
  checked by independent reviewers and numerically over tens of thousands of adversarial runs.
- **Data pipeline.** Binance 5-minute spot bars for BTCUSDT and ETHUSDT (2020–2025), with per-file
  timestamp-unit detection, verified checksums, gap and halt reconciliation, strict no-look-ahead
  features and labels, a sealed test set written once and hashed, and an independent leakage audit.
- **Simulation study.** Regime-shift data-generating processes with exact oracle quantiles, a
  pre-registered hypothesis set, event-level inference for recovery, null-regime diagnostics for
  over-widening, and an equal-budget model-selection rule applied before any real-data comparison.
- **Baselines.** Delayed-feedback ACI and quantile tracking, a faithful port of the multi-step online
  conformal reference implementation with a fidelity harness against the original R package, classical
  volatility models, and a compact TCN trained on Apple-silicon and HPC GPUs.

## How it is run

Pre-registered hypotheses and metrics, a sealed test period, configs and seeds logged with every result,
negative results recorded as results, decision records for every design change, and adversarial review
of proofs and analysis before anything is called a finding. Several early intuitions in this project
were overturned by measurement; the write-up keeps those corrections visible.

## Status (October 2026)

Theory, data pipeline and simulation framework complete; simulation and real-data experiments in
progress. Preliminary simulation evidence: the guarantee holds exactly as proved; most of the measured
recovery gain comes from the pending-outcome accounting rather than from early crediting alone; and the
event definition used to score recovery matters as much as the method. Paper in preparation.

## Stack

Python 3.12 · NumPy · pandas · PyTorch (MPS / CUDA) · statsmodels · SciPy · pytest · SLURM on ISU's Nova cluster · R (reference baseline)
