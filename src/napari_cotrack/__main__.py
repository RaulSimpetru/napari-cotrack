"""Entry point: ``napari-cotrack`` opens napari with the plugin dock visible.

For headless / batch use, the per-stage CLIs still work:
    python -m napari_cotrack.pipeline.track --project ...
    python -m napari_cotrack.pipeline.filter --project ...
    python -m napari_cotrack.pipeline.render --project ...
    python -m napari_cotrack.pipeline.review {extract-all,promote,jumps} --project ...
    python -m napari_cotrack.pipeline.extract anchors --video ... --output ... ...
"""

from __future__ import annotations


def main() -> int:
    import napari
    from napari_cotrack.plugin import NapariCotrackWidget

    viewer = napari.Viewer(title="napari-cotrack")
    widget = NapariCotrackWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="napari-cotrack")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
