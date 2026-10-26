# Neural Forecasting Research

**Online calibration of neural volatility forecasts with early, one-sided feedback.**
Akshat Valse · Iowa State University (Statistics, Mathematics, Data Science, Computer Science) · research project, 2026

This page is a public overview. The code, results, proofs, experiment logs and write-ups remain private
pending publication. I am happy to discuss the methods and my contributions within those limits
([GitHub profile](https://github.com/AkshatValse)).

## The problem

Risk forecasts for a *cumulative* quantity, such as the realized variance of an asset over the next six or
twenty-four hours, are usually scored when the horizon ends. A calibrator that waits for the complete
outcome receives feedback one full horizon later, which can slow its response to a volatility shift.

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
adaptive conformal inference and a multi-step online conformal reference baseline, *without*
systematically widening the bounds the rest of the time.

## What has been built

- **Theory.** A draft long-run coverage guarantee for the calibrator under stated assumptions on early
  crediting, bounded corrections and missing data, with a sharpness construction. Invariant tests and
  adversarial numerical checks test consequences of the proposed argument; numerical agreement does
  not establish the theorem.
- **Data pipeline.** Binance 5-minute spot bars for BTCUSDT and ETHUSDT (2020–2025), with per-file
  timestamp-unit detection, verified checksums, gap and halt reconciliation, strict no-look-ahead
  features and labels, a sealed test set written once and hashed, and automated leakage checks.
- **Simulation framework.** Regime-shift data-generating processes with exact oracle quantiles, a
  pre-registered hypothesis set, event-level inference for recovery, null-regime diagnostics for
  over-widening, and an equal-budget model-selection rule to apply before the full real-data comparison.
  The main simulation study is still in progress.
- **Baselines and models.** Implementations of delayed-feedback ACI and quantile tracking, classical
  volatility models, and a compact TCN, with Apple-silicon and HPC GPU workflows. The Python port of the
  multi-step online conformal reference implementation and its fidelity validation against the original
  R package are in progress; full model training and comparative evaluation remain unfinished.

## How it is run

Pre-registered hypotheses and metrics, a sealed test period, configs and seeds logged with every result,
negative results recorded as results, decision records for every design change, and adversarial review
of proofs and analysis before anything is called a finding. Several early intuitions in this project
were overturned by measurement; the write-up keeps those corrections visible.

## Status (October 2026)

Active research (October 2026). The calibrator, data pipeline and simulation framework are implemented.
The theory is presented as a proof draft; the main simulation study, baseline fidelity validation and
full real-data evaluation are in progress. Preliminary checks inform the study design; final claims
about recovery and comparative performance await the completed evaluation. Paper in preparation.

## Stack

Python 3.12 · NumPy · pandas · PyTorch (MPS / CUDA) · statsmodels · SciPy · pytest · SLURM on ISU's Nova cluster · R (reference baseline)

