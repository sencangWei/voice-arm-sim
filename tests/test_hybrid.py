"""Camera/control contracts only; physical performance is tested separately."""
import unittest

import numpy as np

from vla.hybrid import execution_plan, locate_surface, servo_action, world_cloud


class HybridContracts(unittest.TestCase):
    def test_custom_instruction_is_unscored_and_release_must_follow_grasp(self):
        plan = dict(command={'english': 'put the cup on the plate'},
                    raw='<ans>[(0.2, 0.3), <action>Close Gripper</action>, '
                        '(0.5, 0.6), <action>Open Gripper</action>]</ans>')
        with self.assertRaisesRegex(ValueError, 'unscored'):
            execution_plan(plan)
        prompt, source, release = execution_plan(plan, unscored=True)
        self.assertEqual(prompt, 'put the cup on the plate')
        self.assertEqual(source['x'], .2)
        self.assertEqual(release['y'], .6)
        plan['raw'] = plan['raw'].replace('Close Gripper', 'SWAP').replace('Open Gripper', 'Close Gripper').replace('SWAP', 'Open Gripper')
        with self.assertRaisesRegex(ValueError, 'close followed'):
            execution_plan(plan, unscored=True)

    def test_depth_discontinuity_separates_touching_image_regions(self):
        y, x = np.mgrid[:50, :80]
        cloud = np.stack((x * .001, y * .001, np.full(x.shape, .901)), axis=-1)
        cloud[20:30, 20:30, 2] = .94
        # Adjacent in the image, but physically separate objects. A 2D connected
        # component would merge them and move the inferred center off target.
        cloud[20:30, 30:40, 0] += .1
        cloud[20:30, 30:40, 2] = .94
        target, region, metadata = locate_surface(cloud, {'x': 24/79, 'y': 24/49})
        self.assertEqual(int(region.sum()), 100)
        np.testing.assert_allclose(target, [.0245, .0245, .94])
        self.assertEqual(metadata['tabletop'], .901)
        self.assertFalse(region[24, 35])
        with self.assertRaisesRegex(ValueError, 'No supported surface'):
            locate_surface(cloud, {'x': .9, 'y': .9})

    def test_table_without_object_or_without_plane_is_rejected(self):
        y, x = np.mgrid[:20, :20]
        cloud = np.stack((x * .001, y * .001, np.full(x.shape, .901)), axis=-1)
        with self.assertRaisesRegex(ValueError, 'No supported surface'):
            locate_surface(cloud, {'x': .5, 'y': .5})
        cloud[..., 2] = 2.
        with self.assertRaisesRegex(ValueError, 'No visible tabletop'):
            locate_surface(cloud, {'x': .5, 'y': .5})

    def test_mirrored_depth_and_extrinsic(self):
        depth = np.array([[1., 2., 3.], [4., 5., 6.]])
        intrinsic = np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]])
        transform = np.eye(4)
        transform[:3, 3] = [.1, -.2, .3]
        cloud = world_cloud(depth, intrinsic, transform)
        np.testing.assert_allclose(cloud[0, 0], [1.6, -.95, 3.3])
        np.testing.assert_allclose(cloud[1, 2], [-1.9, .8, 4.3])
        for v, u in np.ndindex(depth.shape):
            camera = cloud[v, u] - transform[:3, 3]
            pixel = intrinsic @ camera
            np.testing.assert_allclose(pixel[:2] / pixel[2], [2-u, v])

    def test_servo_sign_limits_and_hold_rotation(self):
        action = servo_action(np.array([0., 0., 1.]), [.1, -.1, 1.1], -1.)
        np.testing.assert_allclose(action, [.45, -.45, .45, 0., 0., 0., -1.])
        np.testing.assert_allclose(servo_action(np.array([0., 0., 1.]), [0., 0., 1.], 1.),
                                   [0., 0., 0., 0., 0., 0., 1.])

    def test_out_of_workspace_and_nonfinite_fail_closed(self):
        for target in ([.5, 0, 1], [0, 0, .8], [0, 0, 1.4], [np.nan, 0, 1]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                servo_action(np.zeros(3), target, 1.)


if __name__ == '__main__':
    unittest.main()
