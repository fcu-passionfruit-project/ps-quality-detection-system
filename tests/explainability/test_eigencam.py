import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch



from passionfruit.explainability.eigencam import (  # noqa: E402
    CamTestError,
    OUTPUT_FILENAMES,
    activation_colorbar,
    apply_colormap,
    build_parser,
    cam_statistics,
    create_cam_images,
    create_comparison,
    load_cam_dependencies,
    make_occlusions,
    normalize_cam_result,
    signed_diagnostic_maps,
    signed_eigencam_projection,
    resolve_target_layer,
    write_outputs,
)


class _DummyInnerModel:
    def __init__(self):
        self.model = torch.nn.ModuleList(
            (torch.nn.Conv2d(3, 4, 1), torch.nn.ReLU(), torch.nn.Identity())
        )


class _DummyYolo:
    def __init__(self):
        self.model = _DummyInnerModel()


class Yolo26CamTestTests(unittest.TestCase):
    def test_cli_defaults_match_experiment_contract(self):
        args = build_parser().parse_args(
            ["--model", "best.pt", "--image", "test.jpg", "--output", "runs/cam"]
        )
        self.assertEqual(args.imgsz, 1024)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.layer_index, -3)
        self.assertEqual(args.conf, 0.25)

    def test_loads_bundled_cam_without_external_checkout(self):
        from passionfruit.third_party.yolo_cam.eigen_cam import EigenCAM
        cam_class, overlay, root = load_cam_dependencies()
        self.assertIs(cam_class, EigenCAM)
        self.assertTrue(callable(overlay))
        self.assertEqual(root, Path(__file__).resolve().parents[2] / "passionfruit/third_party/yolo_cam")

    def test_resolves_negative_layer_and_rejects_out_of_range_index(self):
        model = _DummyYolo()
        layer, resolved, count = resolve_target_layer(model, -3)
        self.assertIs(layer, model.model.model[0])
        self.assertEqual((resolved, count), (0, 3))
        with self.assertRaisesRegex(CamTestError, "超出範圍"):
            resolve_target_layer(model, 3)

    def test_cam_shape_range_and_arg_coordinates_are_validated(self):
        values = np.zeros((1, 8, 12), dtype=np.float32)
        values[0, 3, 7] = 1.0
        cam = normalize_cam_result(values, (8, 12))
        self.assertEqual(cam_statistics(cam)["argmax_xy"], [7, 3])
        with self.assertRaisesRegex(CamTestError, "原圖尺寸"):
            normalize_cam_result(np.zeros((1, 4, 4), dtype=np.float32), (8, 12))
        with self.assertRaisesRegex(CamTestError, "NaN"):
            normalize_cam_result(np.full((1, 8, 12), np.nan), (8, 12))

    def test_jet_and_grayscale_polarity_are_not_inverted(self):
        cam = np.array([[0.0, 1.0]], dtype=np.float32)
        grayscale = apply_colormap(cam, None)
        jet = apply_colormap(cam, cv2.COLORMAP_JET)
        self.assertEqual(grayscale[0, 0].tolist(), [0, 0, 0])
        self.assertEqual(grayscale[0, 1].tolist(), [255, 255, 255])
        self.assertGreater(int(jet[0, 0, 0]), int(jet[0, 0, 2]))
        self.assertGreater(int(jet[0, 1, 2]), int(jet[0, 1, 0]))
        colorbar = activation_colorbar(256, cv2.COLORMAP_JET)
        self.assertGreater(int(colorbar[45, 5, 0]), int(colorbar[45, 5, 2]))
        self.assertGreater(int(colorbar[45, -6, 2]), int(colorbar[45, -6, 0]))

    def test_signed_projection_and_diagnostics_preserve_both_polarities(self):
        activation = np.array(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[4.0, 1.0], [0.0, 2.0]],
                ]
            ],
            dtype=np.float32,
        )
        signed = signed_eigencam_projection(activation)
        self.assertEqual(signed.shape, (2, 2))
        self.assertLess(float(signed.min()), 0.0)
        self.assertGreater(float(signed.max()), 0.0)
        maps = signed_diagnostic_maps(signed, (4, 4))
        self.assertEqual(set(maps), {"positive", "negative", "absolute"})
        self.assertTrue(all(value.shape == (4, 4) for value in maps.values()))
        self.assertFalse(np.array_equal(maps["positive"], maps["negative"]))
        flipped = signed_diagnostic_maps(-signed, (4, 4))
        np.testing.assert_allclose(maps["positive"], flipped["negative"])
        np.testing.assert_allclose(maps["negative"], flipped["positive"])
        np.testing.assert_allclose(maps["absolute"], flipped["absolute"])

    def test_occlusion_selects_exact_reproducible_high_and_low_ten_percent(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        cam = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
        high, low, high_mask, low_mask = make_occlusions(image, cam)
        self.assertEqual(np.count_nonzero(high_mask), 10)
        self.assertEqual(np.count_nonzero(low_mask), 10)
        self.assertFalse(np.any(high_mask & low_mask))
        self.assertTrue(np.all(high[high_mask] == 114))
        self.assertTrue(np.all(low[low_mask] == 114))
        repeated = make_occlusions(image, cam)
        self.assertTrue(np.array_equal(high_mask, repeated[2]))
        self.assertTrue(np.array_equal(low_mask, repeated[3]))

    def test_outputs_include_numeric_data_and_colorbar_visualizations(self):
        original = np.zeros((48, 64, 3), dtype=np.uint8)
        original[:, :, 2] = 255
        cam = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (48, 1))

        def fake_overlay(image, mask, use_rgb, colormap):
            self.assertFalse(use_rgb)
            self.assertEqual(colormap, cv2.COLORMAP_JET)
            self.assertEqual(image.shape[:2], mask.shape)
            return np.rint(image * 255).astype(np.uint8)

        cam_images = create_cam_images(original, cam, fake_overlay)
        comparison = create_comparison(
            (
                original,
                original,
                cam_images["heatmap_raw"],
                cam_images["overlay_raw"],
            ),
            ("Original", "YOLO Prediction", "EigenCAM Heatmap", "CAM Overlay"),
            "best.pt",
            -3,
            1024,
        )
        image_artifacts = {
            "original": original,
            "grayscale": cam_images["grayscale"],
            "heatmap": cam_images["heatmap"],
            "overlay": cam_images["overlay"],
            "prediction": original,
            "comparison": comparison,
            "gray": cam_images["gray"],
            "jet": cam_images["jet"],
            "turbo": cam_images["turbo"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cam"
            write_outputs(output, image_artifacts, cam, cam - 0.5, {"ok": True})
            self.assertEqual(np.load(output / "cam_normalized.npy").shape, cam.shape)
            self.assertEqual(np.load(output / "cam_signed_raw.npy").shape, cam.shape)
            self.assertEqual(
                json.loads((output / "cam_metrics.json").read_text()), {"ok": True}
            )
            grayscale = cv2.imread(
                str(output / "cam_grayscale.png"), cv2.IMREAD_GRAYSCALE
            )
            self.assertIsNotNone(grayscale)
            self.assertEqual(grayscale.size, original.shape[0] * original.shape[1])
            grayscale = grayscale.reshape(original.shape[:2])
            self.assertEqual(int(grayscale[0, 0]), 0)
            self.assertEqual(int(grayscale[0, -1]), 255)
            self.assertGreater(
                cv2.imread(str(output / "cam_jet.png")).shape[0], original.shape[0]
            )


if __name__ == "__main__":
    unittest.main()
