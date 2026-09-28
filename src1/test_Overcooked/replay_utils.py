import json
import math
import os
import shutil

from project_paths import ensure_local_libs

ensure_local_libs()

from overcooked_ai_py.mdp.actions import Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False


TERRAIN_COLORS = {
    " ": "#F3E9D2",
    "X": "#8D99AE",
    "O": "#E9C46A",
    "T": "#E76F51",
    "D": "#9AD1F5",
    "P": "#5C677D",
    "S": "#7CB518",
}

TERRAIN_LABELS = {
    "O": "Onion",
    "T": "Tomato",
    "D": "Dish",
    "P": "Pot",
    "S": "Serve",
}

PLAYER_COLORS = ["#2A9D8F", "#D62828"]
OBJECT_COLORS = {
    "onion": "#F4D35E",
    "tomato": "#E63946",
    "dish": "#D9EAF4",
    "soup": "#F77F00",
}

ORIENTATION_TO_ARROW = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}


def _build_trajectory_dict(mdp, states, rewards):
    trajectory_states = list(states[: len(rewards)]) if rewards else list(states[:1])
    done_flags = [False] * max(len(rewards) - 1, 0) + ([True] if rewards else [])
    total_return = sum(rewards)
    return {
        "ep_observations": [trajectory_states],
        "ep_actions": [[(Direction.NORTH, Direction.NORTH)] * len(rewards)],
        "ep_rewards": [rewards],
        "ep_dones": [done_flags],
        "ep_returns": [total_return],
        "ep_returns_sparse": [total_return],
        "ep_lengths": [len(rewards)],
        "mdp_params": [mdp.mdp_params],
        "env_params": [{"horizon": len(rewards)}],
    }


def _load_font(size):
    if not PIL_AVAILABLE:
        return None
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _draw_centered_text(draw, box, text, font, fill):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) / 2
    y = top + (bottom - top - text_h) / 2
    draw.text((x, y), text, font=font, fill=fill)


def _object_badge_text(obj):
    if obj.name == "onion":
        return "O"
    if obj.name == "tomato":
        return "T"
    if obj.name == "dish":
        return "D"
    if obj.name == "soup":
        soup_type, num_items, cook_time = obj.state
        del cook_time
        ready_mark = "!" if num_items >= 3 else ""
        soup_initial = "O" if soup_type == "onion" else "T"
        return f"{soup_initial}{num_items}{ready_mark}"
    return obj.name[:1].upper()


def _draw_object_pil(draw, box, obj, tile_font):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    inset_x = width * 0.18
    inset_y = height * 0.18
    badge_box = (
        left + inset_x,
        top + inset_y,
        right - inset_x,
        bottom - inset_y,
    )
    fill = OBJECT_COLORS.get(obj.name, "#FFFFFF")
    draw.rounded_rectangle(badge_box, radius=10, fill=fill, outline="#333333", width=2)
    _draw_centered_text(draw, badge_box, _object_badge_text(obj), tile_font, fill="#111111")


def _draw_player_pil(draw, box, player, player_idx, badge_font):
    left, top, right, bottom = box
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    radius = min(right - left, bottom - top) * 0.28
    color = PLAYER_COLORS[player_idx % len(PLAYER_COLORS)]

    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=color,
        outline="#14213D",
        width=3,
    )
    _draw_centered_text(draw, (cx - radius, cy - radius, cx + radius, cy + radius), f"P{player_idx}", badge_font, fill="#FFFFFF")

    dx, dy = ORIENTATION_TO_ARROW.get(player.orientation, (0, 0))
    arrow_scale = radius * 0.9
    tip = (cx + dx * arrow_scale, cy + dy * arrow_scale)
    side_scale = radius * 0.45
    if dx != 0:
        base1 = (cx - dx * side_scale, cy - side_scale)
        base2 = (cx - dx * side_scale, cy + side_scale)
    else:
        base1 = (cx - side_scale, cy - dy * side_scale)
        base2 = (cx + side_scale, cy - dy * side_scale)
    draw.polygon([tip, base1, base2], fill="#14213D")

    if player.held_object is not None:
        held_box = (
            right - radius * 1.2,
            top + radius * 0.2,
            right - radius * 0.1,
            top + radius * 1.2,
        )
        draw.rounded_rectangle(held_box, radius=8, fill="#FFFFFF", outline="#333333", width=2)
        _draw_centered_text(draw, held_box, _object_badge_text(player.held_object), badge_font, fill="#111111")


def render_state_image(mdp, state, layout, p0_algo, p1_algo, frame_idx, cumulative_reward, tile_size=72):
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not available for raster rendering")

    hud_height = 84
    width = mdp.width * tile_size
    height = mdp.height * tile_size + hud_height

    image = Image.new("RGB", (width, height), "#FAF7F0")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(max(18, tile_size // 4))
    text_font = _load_font(max(14, tile_size // 6))
    tile_font = _load_font(max(16, tile_size // 5))
    badge_font = _load_font(max(12, tile_size // 6))

    draw.rectangle((0, 0, width, hud_height), fill="#1F2937")
    draw.text((18, 12), f"{layout} | P0={p0_algo} | P1={p1_algo}", font=title_font, fill="#FFFFFF")
    draw.text((18, 46), f"Frame {frame_idx:03d} | Score {cumulative_reward}", font=text_font, fill="#D1FAE5")

    grid_top = hud_height
    for y, row in enumerate(mdp.terrain_mtx):
        for x, terrain in enumerate(row):
            left = x * tile_size
            top = grid_top + y * tile_size
            right = left + tile_size
            bottom = top + tile_size
            fill = TERRAIN_COLORS.get(terrain, "#ECECEC")
            draw.rectangle((left, top, right, bottom), fill=fill, outline="#C5C8CE", width=2)

            if terrain in TERRAIN_LABELS:
                label_box = (left + 6, top + tile_size - 26, right - 6, bottom - 6)
                draw.rounded_rectangle(label_box, radius=8, fill="#FFFFFF", outline="#999999", width=1)
                _draw_centered_text(draw, label_box, TERRAIN_LABELS[terrain], badge_font, fill="#333333")

    for obj in state.objects.values():
        x, y = obj.position
        left = x * tile_size
        top = grid_top + y * tile_size
        right = left + tile_size
        bottom = top + tile_size
        _draw_object_pil(draw, (left, top, right, bottom), obj, tile_font)

    for idx, player in enumerate(state.players):
        x, y = player.position
        left = x * tile_size
        top = grid_top + y * tile_size
        right = left + tile_size
        bottom = top + tile_size
        _draw_player_pil(draw, (left, top, right, bottom), player, idx, badge_font)

    return image


def _render_state_matplotlib(
    mdp,
    state,
    layout,
    p0_algo,
    p1_algo,
    frame_idx,
    cumulative_reward,
    frame_path,
    tile_size,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrow, Rectangle

    fig_w = max(6, mdp.width * tile_size / 80)
    fig_h = max(4, mdp.height * tile_size / 80 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_xlim(0, mdp.width)
    ax.set_ylim(-1.2, mdp.height)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0, mdp.height - 0.1), mdp.width, 1.1, color="#1F2937"))
    ax.text(0.15, mdp.height + 0.55, f"{layout} | P0={p0_algo} | P1={p1_algo}", color="white", fontsize=12, va="center")
    ax.text(0.15, mdp.height + 0.2, f"Frame {frame_idx:03d} | Score {cumulative_reward}", color="#D1FAE5", fontsize=10, va="center")

    for y, row in enumerate(mdp.terrain_mtx):
        for x, terrain in enumerate(row):
            plot_y = mdp.height - 1 - y
            ax.add_patch(Rectangle((x, plot_y), 1, 1, facecolor=TERRAIN_COLORS.get(terrain, "#ECECEC"), edgecolor="#C5C8CE", linewidth=1.5))
            if terrain in TERRAIN_LABELS:
                ax.text(x + 0.5, plot_y + 0.12, TERRAIN_LABELS[terrain], fontsize=6, ha="center", va="bottom", color="#333333")

    for obj in state.objects.values():
        x, y = obj.position
        plot_y = mdp.height - 1 - y
        ax.add_patch(Circle((x + 0.5, plot_y + 0.5), 0.22, facecolor=OBJECT_COLORS.get(obj.name, "#FFFFFF"), edgecolor="#333333", linewidth=1.5))
        ax.text(x + 0.5, plot_y + 0.5, _object_badge_text(obj), fontsize=8, ha="center", va="center", color="#111111")

    for idx, player in enumerate(state.players):
        x, y = player.position
        plot_y = mdp.height - 1 - y
        color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
        ax.add_patch(Circle((x + 0.5, plot_y + 0.5), 0.24, facecolor=color, edgecolor="#14213D", linewidth=1.8))
        ax.text(x + 0.5, plot_y + 0.5, f"P{idx}", fontsize=8, ha="center", va="center", color="white")
        dx, dy = ORIENTATION_TO_ARROW.get(player.orientation, (0, 0))
        ax.add_patch(FancyArrow(x + 0.5, plot_y + 0.5, dx * 0.22, -dy * 0.22, width=0.03, head_width=0.13, head_length=0.1, color="#14213D"))
        if player.held_object is not None:
            ax.text(x + 0.78, plot_y + 0.8, _object_badge_text(player.held_object), fontsize=7, ha="center", va="center", color="#111111", bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#333333", "lw": 1})

    fig.savefig(frame_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_episode_replay(
    mdp,
    states,
    rewards,
    output_dir,
    layout,
    p0_algo,
    p1_algo,
    selected_episode,
    fps=6,
    tile_size=72,
    save_frames=True,
    save_gif=True,
):
    os.makedirs(output_dir, exist_ok=True)

    should_materialize_frames = save_frames or save_gif
    frames_dir = os.path.join(output_dir, "frames")
    if should_materialize_frames:
        os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []
    backend = "official_overcooked"
    official_error = None

    if should_materialize_frames:
        try:
            trajectories = _build_trajectory_dict(mdp, states, rewards)
            visualizer = StateVisualizer(tile_size=tile_size)
            visualizer.display_rendered_trajectory(
                trajectories=trajectories,
                trajectory_idx=0,
                img_directory_path=frames_dir,
                img_extension=".png",
                img_prefix="frame_",
                ipython_display=False,
            )
            frame_paths = sorted(
                os.path.join(frames_dir, name)
                for name in os.listdir(frames_dir)
                if name.endswith(".png")
            )
        except Exception as exc:
            official_error = str(exc)
            backend = "pillow" if PIL_AVAILABLE else "matplotlib"
            if not PIL_AVAILABLE:
                save_gif = False
                save_frames = True
            frame_paths = []
            cumulative_reward = 0
            for frame_idx, state in enumerate(states):
                if frame_idx > 0 and frame_idx - 1 < len(rewards):
                    cumulative_reward += rewards[frame_idx - 1]

                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
                if PIL_AVAILABLE:
                    image = render_state_image(
                        mdp=mdp,
                        state=state,
                        layout=layout,
                        p0_algo=p0_algo,
                        p1_algo=p1_algo,
                        frame_idx=frame_idx,
                        cumulative_reward=cumulative_reward,
                        tile_size=tile_size,
                    )
                    image.save(frame_path)
                else:
                    _render_state_matplotlib(
                        mdp=mdp,
                        state=state,
                        layout=layout,
                        p0_algo=p0_algo,
                        p1_algo=p1_algo,
                        frame_idx=frame_idx,
                        cumulative_reward=cumulative_reward,
                        frame_path=frame_path,
                        tile_size=tile_size,
                    )
                frame_paths.append(frame_path)

    gif_path = None
    if save_gif and frame_paths and PIL_AVAILABLE:
        duration_ms = max(20, int(math.ceil(1000 / max(fps, 1))))
        gif_path = os.path.join(output_dir, "replay.gif")
        images = [Image.open(frame_path) for frame_path in frame_paths]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        for image in images:
            image.close()

    if should_materialize_frames and not save_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir = None

    metadata = {
        "layout": layout,
        "p0": p0_algo,
        "p1": p1_algo,
        "selected_episode": selected_episode,
        "frame_count": len(states),
        "frames_dir": frames_dir,
        "gif_path": gif_path,
        "fps": fps,
        "tile_size": tile_size,
        "render_backend": backend,
        "gif_requested": save_gif,
        "pillow_available": PIL_AVAILABLE,
        "official_renderer_error": official_error,
    }

    metadata_path = os.path.join(output_dir, "replay_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    metadata["metadata_path"] = metadata_path
    return metadata
