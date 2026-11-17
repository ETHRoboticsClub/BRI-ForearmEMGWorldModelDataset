import threading

import mujoco
import numpy as np

from femg.hand.geometry import hand_frame, neutral_pose, rotation, unit_vector, visible_bounds
from femg.hand.mapping import HandMap, Rule, label_text, set_pose


class HandScene:
    """A hand in a viser server, posed one frame at a time. switch() changes the hand model."""

    def __init__(self, port: int, host: str = "127.0.0.1", dark: bool = False, name: str = "shadow_hand"):
        import viser

        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.on_client_connect(lambda client: threading.Timer(0.5, self.prepare_client, (client,)).start())
        self.set_theme(dark)
        self.labels_visible = True
        self.switch(name)

    def switch(self, name: str):
        from mjviser import ViserMujocoScene

        self.server.scene.reset()
        self.hand = HandMap(name)
        self.model, self.data, self.ids = self.hand.load()
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)
        self.scene.show_tendons = False  # musculoskeletal hands (MyoHand): show the bones, not the muscles
        add_frame = self.server.scene.add_frame
        groups = [add_frame(name, show_axes=False) for name in ("/bodies", "/hull", "/decor", "/labels")]
        self.frames = [self.scene.fixed_bodies_frame, *groups]
        self.turn = np.eye(3)
        self.point_camera()
        self.pivots = self.group_by_pivot()
        self.labels = [
            self.server.scene.add_label(f"/labels/{i}", "", font_screen_scale=0.8, visible=self.labels_visible)
            for i in range(len(self.pivots))
        ]
        self.hide_panel()

    def point_camera(self):
        """Palm seen from the thumb side, fingers up, the whole model in view and centred: the same for every hand.
        The centre of the model is also the pivot of the trackball, so it stays centred while it turns."""
        s = self.hand.settings
        palmar, radial, distal = hand_frame(self.model, self.data, s["palmar_joint"], s["radial_body"], s["ulnar_body"])
        neutral_pose(self.model, self.data)
        center, radius = visible_bounds(self.model, self.data)
        position = center + 1.7 * radius * unit_vector(1.4 * palmar + 0.6 * radial - 0.3 * distal)
        self.camera = (position, center, distal)
        self.pivot = center
        self.server.initial_camera.position, self.server.initial_camera.look_at = tuple(position), tuple(center)
        self.server.initial_camera.up = tuple(distal)
        for client in self.server.get_clients().values():
            self.aim(client)

    def aim(self, client, attempts: int = 10):
        """Point a viewer at the centre of the hand, retrying while it has not reported its camera yet."""
        position, center, up = self.camera
        try:
            client.camera.up_direction = up
            client.camera.position, client.camera.look_at = position, center
        except AssertionError:
            if attempts > 1:
                threading.Timer(0.3, self.aim, (client, attempts - 1)).start()

    def orbit(self, right: float, up: float):
        """Trackball: turn the hand itself about the viewer's vertical and horizontal axes (rad). The camera stays
        still, so viser's orbit limits (stuck at the poles) never come into play."""
        clients = list(self.server.get_clients().values())
        try:
            camera = clients[0].camera
            forward = unit_vector(np.asarray(camera.look_at) - np.asarray(camera.position))
            top = np.asarray(camera.up_direction, float)
        except (IndexError, AssertionError):
            return
        side = unit_vector(np.cross(forward, top))
        top = np.cross(side, forward)
        self.turn = rotation(top, right) @ rotation(side, up) @ self.turn
        self.apply_turn()

    def apply_turn(self):
        """Rotate the whole hand (all of mjviser's scene groups and the labels) by self.turn about its center."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.turn.flatten())
        shift = self.pivot - self.turn @ self.pivot
        for frame in self.frames:
            frame.wxyz, frame.position = quat, shift

    def zoom(self, factor: float):
        for client in self.server.get_clients().values():
            try:
                target = np.asarray(client.camera.look_at)
                client.camera.position = target + factor * (np.asarray(client.camera.position) - target)
                client.camera.look_at = target
            except AssertionError:
                pass

    def reset_view(self):
        self.turn = np.eye(3)
        self.apply_turn()
        for client in self.server.get_clients().values():
            self.aim(client)

    def prepare_client(self, client):
        self.aim(client)
        client.gui.main_panel.minimize()

    def group_by_pivot(self) -> list[list[Rule]]:
        """Joints that turn about the same point (e.g. MCP spread + flexion) share one label."""
        neutral_pose(self.model, self.data)
        groups: list[tuple[np.ndarray, list[Rule]]] = []
        for r in self.hand.rules:
            if not r.label:
                continue
            anchor = self.data.xanchor[self.ids[r.joint]].copy()
            for position, rules in groups:
                if np.linalg.norm(position - anchor) < 0.004:
                    rules.append(r)
                    break
            else:
                groups.append((anchor, [r]))
        return [rules for _, rules in groups]

    def show(self, kin_row: np.ndarray, unit: bool = False, centers: np.ndarray | None = None):
        set_pose(self.model, self.data, self.hand, self.ids, kin_row, unit, centers)
        how = self.hand.angles(kin_row, unit, centers)
        self.scene.update_from_mjdata(self.data)
        self.scene.fixed_bodies_frame.position = self.pivot - self.turn @ self.pivot  # mjviser resets it on update
        for label, rules in zip(self.labels, self.pivots, strict=True):
            label.position = self.data.xanchor[self.ids[rules[0].joint]]
            label.text = "  ".join(t for r in rules if (t := label_text(r, kin_row, unit, how[r.joint][1])))

    def set_theme(self, dark: bool):
        self.server.gui.configure_theme(dark_mode=dark, show_logo=False, show_share_button=False)
        self.hide_panel()

    def hide_panel(self):
        """The viser control panel is not used: keep it minimized (configure_theme opens it again)."""
        self.server.gui.main_panel.minimize()
        for client in self.server.get_clients().values():
            client.gui.main_panel.minimize()

    def show_labels(self, visible: bool):
        self.labels_visible = visible
        for label in self.labels:
            label.visible = visible
