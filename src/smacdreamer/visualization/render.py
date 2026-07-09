"""Frame capture, overlay, MP4 writing, and an optional interactive window.

Reuses the existing SMAClite Pygame renderer for the battlefield image; this module only
adds a readable text overlay, an MP4 writer (via imageio if available), and a live display
window for interactive playback. Pygame and imageio are imported lazily so importing this
module (and the pure trace helpers alongside it) never requires a display or ffmpeg.

Headless: callers must set ``SDL_VIDEODRIVER=dummy`` / ``SDL_AUDIODRIVER=dummy`` BEFORE the
first pygame initialisation (i.e. before the first render). The visualise scripts do this in
``main()`` when ``--headless`` is passed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def get_raw_smaclite_env(adapter):
    """Return the underlying ``SMACliteEnv`` from a ``SMACliteR2DreamerAdapter``.

    The adapter wraps ``SMACliteDreamerEnv`` which wraps the SMAClite env (possibly behind
    a gym TimeLimit for builtin scenarios). We never edit the env classes; we just reach the
    object that owns ``render()``/``render_mode``.
    """
    sde = adapter.unwrapped                       # SMACliteDreamerEnv
    inner = getattr(sde, "_env", sde)             # smaclite env (maybe gym-wrapped)
    return getattr(inner, "unwrapped", inner)


def enable_rgb_render(adapter) -> None:
    """Switch the underlying SMAClite env to ``rgb_array`` render mode (for frame capture)."""
    get_raw_smaclite_env(adapter).render_mode = "rgb_array"


def capture_frame(adapter, scale: int = 1) -> np.ndarray:
    """Render the current SMAClite state to an ``(H, W, 3)`` uint8 RGB array (a fresh copy).

    ``pygame.surfarray.pixels3d`` (used inside the renderer) returns a locked view of the
    surface, so we copy eagerly to a contiguous array that is safe to keep/append.

    ``scale`` nearest-neighbour upscales each axis (maps are tiny — 32 px/tile — so units and
    overlay text are hard to read at native size).
    """
    raw = get_raw_smaclite_env(adapter)
    if getattr(raw, "render_mode", None) != "rgb_array":
        raw.render_mode = "rgb_array"
    frame = np.ascontiguousarray(np.asarray(raw.render(), dtype=np.uint8))
    if scale and int(scale) > 1:
        s = int(scale)
        frame = np.ascontiguousarray(frame.repeat(s, axis=0).repeat(s, axis=1))
    return frame


def draw_overlay(frame_rgb: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Blit a small black-on-white text overlay (one entry per line) onto an RGB frame.

    Returns a new ``(H, W, 3)`` uint8 array. Uses pygame's font module, which renders fine
    under the SDL dummy video driver (no display needed). On any pygame/font failure the
    original frame is returned unchanged (overlay is best-effort, never fatal).
    """
    if not lines:
        return frame_rgb
    try:
        import pygame

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        # make_surface wants (W, H, 3); our frame is (H, W, 3).
        surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        # Auto-size the font to the frame so text stays readable after upscaling.
        font_size = max(12, int(frame_rgb.shape[0] * 0.032))
        font = pygame.font.SysFont("monospace", font_size)
        y = 2
        for line in lines:
            text = font.render(str(line), True, (0, 0, 0), (255, 255, 255))
            surf.blit(text, (3, y))
            y += text.get_height() + 1
        out = pygame.surfarray.array3d(surf)          # (W, H, 3)
        return np.ascontiguousarray(np.transpose(out, (1, 0, 2)).astype(np.uint8))
    except Exception as exc:  # pragma: no cover - overlay is best-effort
        print(f"[visualize] overlay disabled (pygame font error: {exc})")
        return frame_rgb


def build_overlay_lines(
    *,
    map_name: str,
    seed: int,
    step: int,
    action_labels: Sequence[str],
    enemies_alive,
    allies_alive,
    enemy_hp_damage_this_step,
    target_focus_score,
    done: bool = False,
    battle_won: Optional[bool] = None,
) -> List[str]:
    """Assemble the default overlay text lines for one frame."""
    lines = [f"{map_name}  seed={seed}  step={step}"]
    if done and battle_won is not None:
        lines.append("RESULT: WIN" if battle_won else "RESULT: LOSS")
    lines.append(f"allies={allies_alive}  enemies={enemies_alive}")
    dmg = 0.0 if enemy_hp_damage_this_step is None else float(enemy_hp_damage_this_step)
    lines.append(f"enemy_dmg_step={dmg:.2f}")
    focus = "n/a" if target_focus_score is None else f"{float(target_focus_score):.2f}"
    lines.append(f"target_focus={focus}")
    if action_labels:
        # Wrap actions across lines so a wide team stays readable on a narrow map.
        per_line = 4
        for i in range(0, len(action_labels), per_line):
            chunk = action_labels[i:i + per_line]
            prefix = "acts: " if i == 0 else "      "
            lines.append(prefix + " ".join(f"{i + j}:{lab}" for j, lab in enumerate(chunk)))
    return lines


def with_hold(frames: Sequence[np.ndarray], fps: float,
              hold_last_seconds: float = 0.0, hold_first_seconds: float = 0.0) -> list:
    """Duplicate the first/last frame so playback pauses on the opening and closing state.

    Holding the final frame lets the viewer read the WIN/LOSS outcome instead of it flashing by.
    """
    frames = list(frames)
    if not frames:
        return frames
    lead = [frames[0]] * int(round(max(0.0, hold_first_seconds) * fps))
    tail = [frames[-1]] * int(round(max(0.0, hold_last_seconds) * fps))
    return lead + frames + tail


def write_mp4(path, frames: Sequence[np.ndarray], fps: float) -> None:
    """Write ``frames`` to an MP4 at ``path`` using imageio. Fails clearly if unavailable."""
    if not frames:
        raise ValueError("write_mp4: no frames to write")
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError(
            "Saving MP4 requires the 'imageio' package with an ffmpeg backend. "
            "Install it with:  pip install imageio imageio-ffmpeg\n"
            f"(import failed: {exc})"
        )
    try:
        imageio.mimwrite(str(path), list(frames), fps=float(fps), macro_block_size=16)
    except Exception as exc:
        raise RuntimeError(
            "Failed to encode MP4 — the imageio ffmpeg backend is likely missing. "
            "Install it with:  pip install imageio-ffmpeg\n"
            f"(encode error: {exc})"
        )


class InteractiveWindow:
    """A small pygame display window for live playback of captured RGB frames.

    Only usable with a real display (not headless). The window owns its own pygame display
    surface so it works regardless of the SMAClite renderer's mode (we feed it the same
    overlaid frames we record).
    """

    def __init__(self, width: int, height: int, fps: float, title: str = "R2-Dreamer replay"):
        import pygame

        if not pygame.get_init():
            pygame.init()
        pygame.display.init()
        pygame.display.set_caption(title)
        self._pygame = pygame
        self._screen = pygame.display.set_mode((int(width), int(height)))
        self._clock = pygame.time.Clock()
        self._fps = float(fps)
        self.closed = False

    def show(self, frame_rgb: np.ndarray) -> None:
        if self.closed:
            return
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        self._screen.blit(surf, (0, 0))
        pygame.display.flip()
        self._clock.tick(self._fps)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._pygame.display.quit()
        except Exception:
            pass
        self.closed = True


__all__ = [
    "get_raw_smaclite_env",
    "enable_rgb_render",
    "capture_frame",
    "draw_overlay",
    "build_overlay_lines",
    "with_hold",
    "write_mp4",
    "InteractiveWindow",
]
