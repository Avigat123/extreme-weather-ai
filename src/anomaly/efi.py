"""
Extreme Forecast Index (EFI)
============================
Measures how unusual a forecast value is relative to the model's own
climatological distribution for that location, variable, and time of year.

EFI in [-1, 1]:
    +1  -> forecast is far above everything ever seen in climatology (extreme high)
    -1  -> forecast is far below everything ever seen in climatology (extreme low)
     0  -> forecast sits right in the middle of the climatological spread

This is the ECMWF-style definition, computed with the climatological CDF
instead of raw ensemble members (works fine with deterministic / AI forecasts
like GraphCast/GenCast/Pangu output, as long as you have a climatology to
compare against).

    EFI = (2 / pi) * integral_0^1  [ F_clim^-1(p) <= F_fcst^-1(p) ? +1 : -1 ] dp

We approximate the integral numerically over p in (0, 1).
"""

from __future__ import annotations
import numpy as np


def _sorted_quantiles(samples: np.ndarray, n_quantiles: int = 101) -> np.ndarray:
    """Return the empirical quantile function of `samples` at n_quantiles points."""
    probs = np.linspace(0.001, 0.999, n_quantiles)
    return np.quantile(samples, probs)


def compute_efi(
    forecast_value: np.ndarray,
    climatology_samples: np.ndarray,
    n_quantiles: int = 101,
) -> np.ndarray:
    """
    Parameters
    ----------
    forecast_value : np.ndarray, shape (H, W)
        Single forecast field for one variable at one lead time (e.g. rain in mm).
    climatology_samples : np.ndarray, shape (N, H, W)
        N historical values (same day-of-year window, same variable) per grid cell,
        e.g. N = 30 years x +/-7 day window = ~450 samples.
    n_quantiles : int
        Resolution of the numerical integration.

    Returns
    -------
    efi : np.ndarray, shape (H, W), values in [-1, 1]
    """
    H, W = forecast_value.shape
    probs = np.linspace(0.001, 0.999, n_quantiles)

    # climatological quantile function per cell: (n_quantiles, H, W)
    clim_q = np.quantile(climatology_samples, probs, axis=0)

    # sign(1 if forecast >= clim quantile at that p else -1), integrated over p
    fcst_bc = forecast_value[None, :, :]  # (1, H, W)
    sign = np.where(fcst_bc >= clim_q, 1.0, -1.0)  # (n_quantiles, H, W)

    efi = (2.0 / np.pi) * np.trapz(sign, probs, axis=0)
    return np.clip(efi, -1.0, 1.0)


def compute_efi_timeseries(
    forecast_sequence: np.ndarray,
    climatology_samples: np.ndarray,
    n_quantiles: int = 101,
) -> np.ndarray:
    """
    Vectorised EFI over a lead-time sequence.

    forecast_sequence : (T, H, W)  forecast at T lead times
    climatology_samples : (N, H, W) same climatology used for every lead time
                           (swap this per lead time if you have day-specific climo)

    Returns
    -------
    efi_seq : (T, H, W)
    """
    T = forecast_sequence.shape[0]
    return np.stack(
        [compute_efi(forecast_sequence[t], climatology_samples, n_quantiles) for t in range(T)],
        axis=0,
    )


def sot_score(forecast_value: np.ndarray, climatology_samples: np.ndarray, tail: str = "upper") -> np.ndarray:
    """
    Shift Of Tail (SOT) — a companion metric to EFI. Tells you not just that the
    forecast is extreme, but that it is more extreme than anything in the
    climatological record (useful for flagging "beyond historical" events).

    SOT > 0 means the forecast exceeds the climatological extreme.
    """
    if tail == "upper":
        clim_extreme = np.quantile(climatology_samples, 0.99, axis=0)
        clim_p90 = np.quantile(climatology_samples, 0.90, axis=0)
    else:
        clim_extreme = np.quantile(climatology_samples, 0.01, axis=0)
        clim_p90 = np.quantile(climatology_samples, 0.10, axis=0)

    denom = np.where(np.abs(clim_extreme - clim_p90) < 1e-6, 1e-6, clim_extreme - clim_p90)
    sot = (forecast_value - clim_extreme) / denom
    return sot
