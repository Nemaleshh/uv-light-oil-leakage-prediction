"""
presence_detector.py  — v2
Motion + background subtraction based car presence detection.

WHY this is better than brightness thresholding
───────────────────────────────────────────────
  • UV light changes brightness when car arrives  ← brightness lies
  • Sunlight floods in when car leaves           ← brightness lies
  • This detector looks at CHANGE vs BACKGROUND  ← never lies

How it works
────────────
  1. MOTION CHECK      : frame-to-frame abs-diff > threshold  → car is moving
  2. STABILITY CHECK   : N consecutive low-motion frames → scene settled
  3. BACKGROUND CHECK  : settled frame vs "empty bay" model → car still there?
  4. HOLD TIMER        : 30 s of stable + car present → trigger capture
  5. DEPARTURE         : after capture, watch for motion + scene returns to
                         empty-bay background → car has left → reset

States
──────
  IDLE          – empty bay, background model is being updated
  MOTION        – high frame-diff, car entering or leaving
  STABILIZING   – motion just stopped, verifying car presence
  DETECTING     – car confirmed present, hold timer counting
  READY         – timer fired, emit trigger (transitions immediately)
  WAITING_LEAVE – capture done, waiting for car to depart
"""

import time
import cv2
import numpy as np


class PresenceState:
    IDLE          = "IDLE"
    MOTION        = "MOTION"
    STABILIZING   = "STABILIZING"
    DETECTING     = "DETECTING"
    READY         = "READY"
    WAITING_LEAVE = "WAITING_LEAVE"


class PresenceDetector:
    """
    Parameters
    ----------
    hold_seconds     : seconds the car must stay still before capture fires
    motion_thresh    : mean per-pixel difference (0-255) that counts as motion
                       lower = more sensitive  |  default 8.0
    presence_thresh  : fraction of pixels that must differ from background
                       to conclude a car is present  |  default 0.20 (20%)
    bg_pixel_diff    : per-pixel tolerance vs background  |  default 25 (of 255)
    stabilize_frames : consecutive still frames needed before presence check
    """

    # Working resolution (downscaled for speed – detection doesn't need full res)
    WORK_W = 320
    WORK_H = 240

    def __init__(
        self,
        hold_seconds:     int   = 30,
        motion_thresh:    float = 8.0,
        presence_thresh:  float = 0.20,
        bg_pixel_diff:    float = 25.0,
        stabilize_frames: int   = 15,
    ):
        self.hold_seconds     = hold_seconds
        self.motion_thresh    = motion_thresh
        self.presence_thresh  = presence_thresh
        self.bg_pixel_diff    = bg_pixel_diff
        self.stabilize_frames = stabilize_frames

        # Background model (float32 grayscale at WORK resolution)
        self._background: np.ndarray | None = None
        self._bg_frames  = 0          # frames accumulated so far
        self._bg_needed  = 60         # frames required before model is trusted

        # Per-frame state
        self._prev_gray: np.ndarray | None = None
        self._stable_count   = 0
        self._detect_start: float | None = None

        self._state = PresenceState.IDLE

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> str:
        return self._state

    @property
    def bg_ready(self) -> bool:
        """True once a reliable background model has been built."""
        return self._bg_frames >= self._bg_needed

    def calibrate_now(self, frame: np.ndarray) -> None:
        """
        Force-set background from the given frame (no car in view).
        Call from UI "Calibrate Background" button.
        """
        gray = self._to_gray(frame)
        self._background  = gray.astype(np.float32)
        self._bg_frames   = self._bg_needed        # mark ready immediately
        self._state       = PresenceState.IDLE
        self._detect_start = None
        self._stable_count = 0

    def reset(self) -> None:
        """Force-reset to IDLE (keeps background model intact)."""
        self._state        = PresenceState.IDLE
        self._detect_start = None
        self._stable_count = 0

    def process(self, frame: np.ndarray) -> tuple[str, int, bool]:
        """
        Analyse one camera frame.

        Returns
        -------
        state        : str  – current PresenceState constant
        seconds_held : int  – seconds car has been confirmed & stable (0 otherwise)
        triggered    : bool – True exactly once when the 30 s hold is reached
        """
        gray   = self._to_gray(frame)
        motion = self._is_motion(gray)
        self._prev_gray = gray

        # Always update background model when genuinely idle & no motion
        if self._state == PresenceState.IDLE and not motion:
            self._update_bg(gray)

        # ── State machine ───────────────────────────────────────────── #

        if self._state == PresenceState.IDLE:
            if motion:
                self._state = PresenceState.MOTION
                self._stable_count = 0
            return self._state, 0, False

        # ── MOTION: car is entering/exiting ─────────────────────────── #
        if self._state == PresenceState.MOTION:
            if motion:
                self._stable_count = 0
            else:
                self._stable_count += 1
                if self._stable_count >= self.stabilize_frames:
                    self._stable_count = 0
                    self._state = PresenceState.STABILIZING
            return self._state, 0, False

        # ── STABILIZING: motion just stopped ────────────────────────── #
        if self._state == PresenceState.STABILIZING:
            if motion:
                # Was a brief stop, still moving
                self._state = PresenceState.MOTION
                self._stable_count = 0
                return self._state, 0, False
            if self._car_is_present(gray):
                # Car confirmed → begin hold timer
                self._state = PresenceState.DETECTING
                self._detect_start = time.time()
            else:
                # Scene returned to background → car just drove past
                self._state = PresenceState.IDLE
            return self._state, 0, False

        # ── DETECTING: car is parked, counting down ──────────────────── #
        if self._state == PresenceState.DETECTING:
            if motion:
                # Something moved → restart from MOTION
                self._state = PresenceState.MOTION
                self._detect_start = None
                self._stable_count = 0
                return self._state, 0, False
            if not self._car_is_present(gray):
                # Car disappeared without moving? (edge case) → back to idle
                self._state = PresenceState.IDLE
                self._detect_start = None
                return self._state, 0, False
            secs = int(time.time() - self._detect_start)
            if secs >= self.hold_seconds:
                self._state = PresenceState.READY
                return self._state, secs, True   # ← FIRE CAPTURE
            return self._state, secs, False

        # ── READY: trigger was just emitted ─────────────────────────── #
        if self._state == PresenceState.READY:
            self._state = PresenceState.WAITING_LEAVE
            return self._state, 0, False

        # ── WAITING_LEAVE: result shown, waiting for car to depart ───── #
        if self._state == PresenceState.WAITING_LEAVE:
            if motion:
                # Car started moving
                self._stable_count = 0
            else:
                self._stable_count += 1
                if self._stable_count >= self.stabilize_frames:
                    # Settled → check if car is actually gone
                    if not self._car_is_present(gray):
                        # ✅ Car is gone — update background and reset
                        self._update_bg(gray)  # recalibrate with current empty bay
                        self._state = PresenceState.IDLE
                        self._stable_count = 0
                        self._detect_start = None
                    else:
                        # Still there (e.g. operator is doing something)
                        self._stable_count = 0   # keep waiting
            return self._state, 0, False

        return PresenceState.IDLE, 0, False

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (self.WORK_W, self.WORK_H))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _is_motion(self, gray: np.ndarray) -> bool:
        """True if mean absolute frame-difference exceeds motion_thresh."""
        if self._prev_gray is None:
            return False
        diff = cv2.absdiff(gray, self._prev_gray)
        # Blur to reduce sensor noise before measuring motion magnitude
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        return float(np.mean(diff)) > self.motion_thresh

    def _update_bg(self, gray: np.ndarray) -> None:
        """Exponential moving average background update."""
        f32 = gray.astype(np.float32)
        if self._background is None:
            self._background = f32.copy()
            self._bg_frames  = 1
        else:
            # Use faster alpha until model is ready, then slow drift
            alpha = 0.30 if not self.bg_ready else 0.03
            cv2.accumulateWeighted(f32, self._background, alpha)
            self._bg_frames = min(self._bg_frames + 1, self._bg_needed * 10)

    def _car_is_present(self, gray: np.ndarray) -> bool:
        """
        Compare current frame to background model, robust to auto-exposure.
        True if ≥ presence_thresh fraction of pixels differ by ≥ bg_pixel_diff.
        """
        if self._background is None or not self.bg_ready:
            return True   # No reference → assume car present (fail-open)

        bg = np.clip(self._background, 0, 255).astype(np.uint8)
        
        # Compensate for global lighting/exposure shifts (bounded to 30 units max)
        gray_mean = float(np.mean(gray))
        bg_mean = float(np.mean(bg))
        shift = np.clip(bg_mean - gray_mean, -30.0, 30.0)
        adjusted_gray = np.clip(gray.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        
        diff = cv2.absdiff(adjusted_gray, bg)
        # Gentle blur to suppress single noisy pixels
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        frac_changed = float(np.mean(diff > self.bg_pixel_diff))
        
        return frac_changed > self.presence_thresh
